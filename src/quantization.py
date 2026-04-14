#!/usr/bin/env python3
"""
Quantization module for implementing FP16, INT8, INT4, and 1-bit (W1.58A8 BitNet-style) quantization.
Supports layer-by-layer sensitivity analysis.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class QuantizationLevel(Enum):
    """Quantization precision levels"""
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    INT1 = "1bit"  # W1.58A8 BitNet-style


class Quantizer:
    """
    Implements various quantization strategies for neural network weights.
    Supports FP16, INT8, INT4, and 1-bit (W1.58A8) quantization.
    """
    
    def __init__(self, level: QuantizationLevel = QuantizationLevel.FP16):
        self.level = level
    
    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Quantize a weight tensor to the specified precision level.
        
        Args:
            weight: Input weight tensor (FP32 or FP16)
            
        Returns:
            Quantized weight tensor
        """
        if self.level == QuantizationLevel.FP16:
            return self._quantize_fp16(weight)
        elif self.level == QuantizationLevel.INT8:
            return self._quantize_int8(weight)
        elif self.level == QuantizationLevel.INT4:
            return self._quantize_int4(weight)
        elif self.level == QuantizationLevel.INT1:
            return self._quantize_1bit(weight)
        else:
            raise ValueError(f"Unknown quantization level: {self.level}")
    
    def _quantize_fp16(self, weight: torch.Tensor) -> torch.Tensor:
        """Convert to FP16 precision"""
        return weight.half()
    
    def _quantize_int8(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Quantize to INT8 using symmetric per-tensor quantization.
        Scale = max(|W|) / 127. Range: [-128, 127].
        """
        scale = weight.abs().max() / 127.0
        q = (weight / scale).round().clamp(-128, 127)
        return (q * scale).to(weight.dtype)
    
    def _quantize_int4(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Quantize to INT4 using symmetric per-tensor quantization.
        Scale = max(|W|) / 7. Range: [-8, 7].
        """
        scale = weight.abs().max() / 7.0
        q = (weight / scale).round().clamp(-8, 7)
        return (q * scale).to(weight.dtype)
    
    def _quantize_1bit(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Quantize to 1-bit (W1.58A8 BitNet-style).
        Weights are mapped to ternary values {-1, 0, +1} scaled by mean absolute value.

        Note: "W1.58" refers to log2(3) ≈ 1.585 bits of entropy in a balanced ternary
        distribution — it is NOT a scaling multiplier. The scale factor is the mean
        absolute value of the weights, following the BitNet b1.58 paper.

        Reference: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
        """
        scale = weight.abs().mean() + 1e-8
        ternary = (weight / scale).round().clamp(-1, 1)
        return (ternary * scale).to(weight.dtype)
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Quantize all linear layers in a model.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Quantized model (in-place modification)
        """
        logger.info(f"Quantizing model to {self.level.value}...")
        
        quantized_count = 0
        total_count = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_count += 1
                original_weight = module.weight.data.clone()
                
                # Quantize weight
                quantized_weight = self.quantize_weight(original_weight)
                
                # Replace weight
                module.weight.data = quantized_weight
                quantized_count += 1
        
        logger.info(f"Quantized {quantized_count}/{total_count} linear layers")
        
        return model
    
    def get_memory_ratio(self) -> float:
        """
        Get memory compression ratio compared to FP32.
        
        Returns:
            Memory ratio (e.g., 0.5 for FP16, 0.25 for INT8, etc.)
        """
        ratios = {
            QuantizationLevel.FP16: 0.5,  # 16-bit vs 32-bit
            QuantizationLevel.INT8: 0.25,  # 8-bit vs 32-bit
            QuantizationLevel.INT4: 0.125,  # 4-bit vs 32-bit
            QuantizationLevel.INT1: 0.03125,  # ~1-bit vs 32-bit (approx)
        }
        return ratios[self.level]


def apply_quantization_to_model(model: nn.Module, quantizer: "Quantizer") -> nn.Module:
    """
    Apply quantization to all linear layers in a model in-place.

    Args:
        model: PyTorch model to quantize
        quantizer: Quantizer instance specifying the precision level

    Returns:
        The same model with all linear layer weights replaced by quantized versions.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            original_dtype = module.weight.data.dtype
            original_device = module.weight.data.device
            quantized_weight = quantizer.quantize_weight(module.weight.data)
            module.weight.data = quantized_weight.to(dtype=original_dtype, device=original_device)
    return model


def apply_quantization_to_layer(
    layer: nn.Module,
    level: QuantizationLevel,
    layer_idx: Optional[int] = None
) -> Tuple[nn.Module, Dict]:
    """
    Apply quantization to a single layer and return metrics.
    
    Args:
        layer: Neural network layer (Linear)
        level: Quantization level
        layer_idx: Optional layer index for logging
        
    Returns:
        Tuple of (quantized_layer, metrics_dict)
    """
    quantizer = Quantizer(level)
    
    # Get original weight stats
    weight = layer.weight.data
    original_mean = weight.mean().item()
    original_std = weight.std().item()
    original_norm = weight.norm().item()
    
    # Quantize
    quantized_weight = quantizer.quantize_weight(weight)
    
    # Compute quantization error
    error = (weight - quantized_weight).abs().mean().item()
    relative_error = error / (original_norm + 1e-6)
    
    # Compute correlation (how well quantized preserves original)
    correlation = torch.nn.functional.cosine_similarity(
        weight.flatten().unsqueeze(0),
        quantized_weight.flatten().unsqueeze(0)
    ).item()
    
    # Replace weight
    layer.weight.data = quantized_weight
    
    metrics = {
        'layer_idx': layer_idx,
        'level': level.value,
        'original_mean': original_mean,
        'original_std': original_std,
        'original_norm': original_norm,
        'quantized_mean': quantized_weight.mean().item(),
        'quantized_std': quantized_weight.std().item(),
        'absolute_error': error,
        'relative_error': relative_error,
        'correlation': correlation,
        'memory_ratio': quantizer.get_memory_ratio()
    }
    
    return layer, metrics


def get_quantization_levels() -> List[QuantizationLevel]:
    """Return all quantization levels for benchmarking"""
    return [
        QuantizationLevel.FP16,
        QuantizationLevel.INT8,
        QuantizationLevel.INT4,
        QuantizationLevel.INT1
    ]


def main():
    """Test quantization module"""
    logger.info("=" * 60)
    logger.info("Quantization Module Test")
    logger.info("=" * 60)
    
    # Create test linear layer
    test_layer = nn.Linear(512, 512)
    logger.info(f"Test layer: {test_layer.in_features} -> {test_layer.out_features}")
    
    # Test all quantization levels
    for level in get_quantization_levels():
        logger.info(f"\nTesting {level.value}...")
        
        # Clone layer to avoid cumulative effects
        layer_copy = nn.Linear(512, 512)
        layer_copy.weight.data = test_layer.weight.data.clone()
        
        # Apply quantization
        _, metrics = apply_quantization_to_layer(layer_copy, level)
        
        logger.info(f"  Memory ratio: {metrics['memory_ratio']}")
        logger.info(f"  Absolute error: {metrics['absolute_error']:.6f}")
        logger.info(f"  Relative error: {metrics['relative_error']:.6f}")
        logger.info(f"  Correlation: {metrics['correlation']:.6f}")
    
    logger.info("\n✓ Quantization module test completed!")


if __name__ == "__main__":
    main()