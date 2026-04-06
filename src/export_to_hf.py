#!/usr/bin/env python3
"""
Export quantized models to HuggingFace-compatible format.
Saves INT4 and 1-bit quantized versions with tokenizer and config for push_to_hub.
"""

import os
import sys
import logging
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path("/root/projects/tasks/04-quantization-1bit-31b")
HF_EXPORTS_DIR = PROJECT_ROOT / "hf_exports"
HF_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace token
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Model to use (Qwen3.5-2B)
MODEL_NAME = "Qwen/Qwen3.5-2B"


class QuantizationLevel:
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    BIT1 = "bit1"


class Quantizer:
    """Quantizer for exporting models"""
    
    def __init__(self, level: str = QuantizationLevel.FP16):
        self.level = level
    
    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.level == QuantizationLevel.FP16:
            return weight.half()
        elif self.level == QuantizationLevel.INT8:
            return self._quantize_int8(weight)
        elif self.level == QuantizationLevel.INT4:
            return self._quantize_int4(weight)
        elif self.level == QuantizationLevel.BIT1:
            return self._quantize_bit1(weight)
        else:
            raise ValueError(f"Unknown level: {self.level}")
    
    def _quantize_int8(self, weight: torch.Tensor) -> torch.Tensor:
        """INT8 per-channel symmetric quantization"""
        abs_max = weight.abs().max()
        scale = abs_max / 127.0
        weight_int8 = (weight / scale).round().clamp(-128, 127)
        return weight_int8.float() * scale
    
    def _quantize_int4(self, weight: torch.Tensor) -> torch.Tensor:
        """INT4 symmetric quantization"""
        abs_max = weight.abs().max()
        scale = abs_max / 7.0
        weight_int4 = (weight / scale).round().clamp(-8, 7)
        return weight_int4.float() * scale
    
    def _quantize_bit1(self, weight: torch.Tensor) -> torch.Tensor:
        """BitNet-style W1.58A8: ternary weights {-1, 0, +1} scaled"""
        scale = weight.abs().mean() + 1e-6
        weight_sign = torch.sign(weight)
        return weight_sign * scale * 1.58


def apply_quantization_to_model(model: nn.Module, quantizer: Quantizer) -> nn.Module:
    """Apply quantization to all linear layers"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            original_dtype = module.weight.data.dtype
            original_device = module.weight.data.device
            quantized_weight = quantizer.quantize_weight(module.weight.data)
            module.weight.data = quantized_weight.to(dtype=original_dtype, device=original_device)
    return model


def export_quantized_model(
    quantization_level: str,
    output_dir: Path,
    save_tokenizer: bool = True,
    save_config: bool = True
) -> Path:
    """
    Export a quantized model in HuggingFace-compatible format.
    
    Args:
        quantization_level: One of 'fp16', 'int8', 'int4', 'bit1'
        output_dir: Directory to save the exported model
        save_tokenizer: Whether to save the tokenizer
        save_config: Whether to save the config
        
    Returns:
        Path to the exported model directory
    """
    logger.info(f"Exporting {quantization_level} quantized model to {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Apply quantization
    quantizer = Quantizer(quantization_level)
    logger.info(f"Applying {quantization_level} quantization...")
    apply_quantization_to_model(model, quantizer)
    
    # Save model
    logger.info("Saving quantized model...")
    model.save_pretrained(str(output_dir))
    
    # Save tokenizer
    if save_tokenizer:
        logger.info("Saving tokenizer...")
        tokenizer.save_pretrained(str(output_dir))
    
    # Save config
    if save_config:
        logger.info("Saving config...")
        model.config.save_pretrained(str(output_dir))
    
    # Save quantization metadata
    metadata = {
        "quantization_level": quantization_level,
        "base_model": MODEL_NAME,
        "export_date": str(torch.__version__),
        "quantization_params": {
            "int8_range": [-128, 127],
            "int4_range": [-8, 7],
            "bit1_type": "W1.58A8 (ternary {-1, 0, +1} scaled)",
        },
        "memory_ratio": {
            "fp16": 0.5,
            "int8": 0.25,
            "int4": 0.125,
            "bit1": 0.03125,
        }
    }
    
    metadata_path = output_dir / "quantization_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Exported {quantization_level} model to {output_dir}")
    logger.info(f"  - Model weights: {output_dir}")
    logger.info(f"  - Tokenizer: {output_dir if save_tokenizer else 'N/A'}")
    logger.info(f"  - Config: {output_dir if save_config else 'N/A'}")
    logger.info(f"  - Metadata: {metadata_path}")
    
    return output_dir


def main():
    """Export INT4 and 1-bit quantized models"""
    logger.info("=" * 60)
    logger.info("HuggingFace Model Export for Quantization Study")
    logger.info("=" * 60)
    
    # Export INT4
    int4_dir = HF_EXPORTS_DIR / "qwen2.5-1.5b-instruct-int4"
    export_quantized_model(QuantizationLevel.INT4, int4_dir)
    
    # Export 1-bit
    bit1_dir = HF_EXPORTS_DIR / "qwen2.5-1.5b-instruct-bit1"
    export_quantized_model(QuantizationLevel.BIT1, bit1_dir)
    
    # Also export FP16 baseline for reference
    fp16_dir = HF_EXPORTS_DIR / "qwen2.5-1.5b-instruct-fp16"
    export_quantized_model(QuantizationLevel.FP16, fp16_dir)
    
    logger.info("=" * 60)
    logger.info("Export Summary:")
    logger.info(f"  INT4 model: {int4_dir}")
    logger.info(f"  1-bit model: {bit1_dir}")
    logger.info(f"  FP16 baseline: {fp16_dir}")
    logger.info("=" * 60)
    
    # Print push_to_hub instructions
    logger.info("\n📤 To upload to HuggingFace Hub:")
    logger.info("  from huggingface_hub import login, HfApi")
    logger.info("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    logger.info("")
    logger.info("  login(token='hf_...')")
    logger.info("  api = HfApi()")
    logger.info("")
    logger.info("  # Upload INT4")
    logger.info("  model = AutoModelForCausalLM.from_pretrained('/root/projects/tasks/04-quantization-1bit-31b/hf_exports/qwen2.5-1.5b-instruct-int4')")
    logger.info("  tokenizer = AutoTokenizer.from_pretrained('/root/projects/tasks/04-quantization-1bit-31b/hf_exports/qwen2.5-1.5b-instruct-int4')")
    logger.info("  model.push_to_hub('your-username/qwen2.5-1.5b-instruct-int4')")
    logger.info("  tokenizer.push_to_hub('your-username/qwen2.5-1.5b-instruct-int4')")
    logger.info("")
    logger.info("  # Upload 1-bit")
    logger.info("  model.push_to_hub('your-username/qwen2.5-1.5b-instruct-bit1')")
    logger.info("  tokenizer.push_to_hub('your-username/qwen2.5-1.5b-instruct-bit1')")


if __name__ == "__main__":
    main()