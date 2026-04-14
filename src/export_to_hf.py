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
PROJECT_ROOT = Path(__file__).parent.parent
HF_EXPORTS_DIR = PROJECT_ROOT / "hf_exports"
HF_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace token
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Model under study
MODEL_NAME = "google/gemma-4-E2B-it"

# Import canonical quantization logic — single source of truth
sys.path.insert(0, str(Path(__file__).parent))
from quantization import Quantizer, QuantizationLevel, apply_quantization_to_model


def export_quantized_model(
    quantization_level: QuantizationLevel,
    output_dir: Path,
    save_tokenizer: bool = True,
    save_config: bool = True
) -> Path:
    """
    Export a quantized model in HuggingFace-compatible format.

    Args:
        quantization_level: QuantizationLevel enum value
        output_dir: Directory to save the exported model
        save_tokenizer: Whether to save the tokenizer
        save_config: Whether to save the config

    Returns:
        Path to the exported model directory
    """
    logger.info(f"Exporting {quantization_level.value} quantized model to {output_dir}")

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
    logger.info(f"Applying {quantization_level.value} quantization...")
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
        "quantization_level": quantization_level.value,
        "base_model": MODEL_NAME,
        "export_date": __import__("time").strftime("%Y-%m-%d %H:%M:%S UTC", __import__("time").gmtime()),
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
    
    logger.info(f"✓ Exported {quantization_level.value} model to {output_dir}")
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
    int4_dir = HF_EXPORTS_DIR / "gemma-4-E2B-it-int4"
    export_quantized_model(QuantizationLevel.INT4, int4_dir)

    # Export 1-bit
    bit1_dir = HF_EXPORTS_DIR / "gemma-4-E2B-it-bit1"
    export_quantized_model(QuantizationLevel.INT1, bit1_dir)

    # Also export FP16 baseline for reference
    fp16_dir = HF_EXPORTS_DIR / "gemma-4-E2B-it-fp16"
    export_quantized_model(QuantizationLevel.FP16, fp16_dir)

    logger.info("=" * 60)
    logger.info("Export Summary:")
    logger.info(f"  INT4 model: {int4_dir}")
    logger.info(f"  1-bit model: {bit1_dir}")
    logger.info(f"  FP16 baseline: {fp16_dir}")
    logger.info("=" * 60)

    logger.info("\nTo upload to HuggingFace Hub:")
    logger.info("  from huggingface_hub import login")
    logger.info("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    logger.info("")
    logger.info("  login(token='hf_...')")
    logger.info("")
    logger.info("  # Upload INT4")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{int4_dir}')")
    logger.info(f"  tokenizer = AutoTokenizer.from_pretrained('{int4_dir}')")
    logger.info("  model.push_to_hub('your-username/gemma-4-E2B-it-int4')")
    logger.info("  tokenizer.push_to_hub('your-username/gemma-4-E2B-it-int4')")
    logger.info("")
    logger.info("  # Upload 1-bit")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{bit1_dir}')")
    logger.info("  model.push_to_hub('your-username/gemma-4-E2B-it-bit1')")


if __name__ == "__main__":
    main()