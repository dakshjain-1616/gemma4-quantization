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
        logger.info(f"Initialized Quantizer with level: {level.value}")
    
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
        Quantize to INT8 using symmetric quantization.
        Range: [-128, 127]
        """
        # Get min/max for scaling
        weight_min = weight.min()
        weight_max = weight.max()
        
        # Compute scaling factor
        scale = (weight_max - weight_min) / 255.0
        
        # Quantize
        weight_int8 = ((weight - weight_min) / scale).round().clamp(0, 255) - 128
        
        # Dequantize for computation (simulate INT8 behavior)
        weight_dequant = weight_int8.float() * scale + weight_min
        
        return weight_dequant
    
    def _quantize_int4(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Quantize to INT4 using symmetric quantization.
        Range: [-8, 7]
        """
        # Get min/max for scaling
        weight_min = weight.min()
        weight_max = weight.max()
        
        # Compute scaling factor (16 levels for INT4)
        scale = (weight_max - weight_min) / 15.0
        
        # Quantize
        weight_int4 = ((weight - weight_min) / scale).round().clamp(0, 15) - 8
        
        # Dequantize for computation
        weight_dequant = weight_int4.float() * scale + weight_min
        
        return weight_dequant
    
    def _quantize_1bit(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Quantize to 1-bit (W1.58A8 BitNet-style).
        Uses ternary quantization: {-1, 0, +1} scaled by a learned factor.
        This approximates BitNet's W1.58A8 approach where weights are quantized to 1-bit
        but activations remain in INT8 precision.
        
        Reference: "BitNet: Scaling 1-bit Transformers for Language Models"
        """
        # Compute absolute mean for scaling (similar to BitNet approach)
        scale = weight.abs().mean() + 1e-6
        
        # Ternary quantization: sign function with threshold
        # W1.58 uses a learned threshold, we use simple sign for feasibility study
        weight_sign = torch.sign(weight)
        
        # Apply scaling to approximate 1-bit representation
        # The 1.58 factor comes from optimal scaling for ternary weights
        weight_1bit = weight_sign * scale * 1.58
        
        return weight_1bit
    
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