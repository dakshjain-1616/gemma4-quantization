#!/usr/bin/env python3
"""
Benchmark evaluation script for quantization levels.
Measures perplexity, memory, and speed on WikiText-2 benchmark.
"""

import os
import sys
import logging
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from datasets import load_dataset

# Import quantization module
from quantization import Quantizer, QuantizationLevel, apply_quantization_to_layer, get_quantization_levels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path("/root/projects/tasks/04-quantization-1bit-31b")
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace token
HF_TOKEN = os.getenv("HF_TOKEN", "")


def load_wikitext2_test(max_samples: int = 100) -> List[str]:
    """Load WikiText-2 test dataset."""
    logger.info(f"Loading WikiText-2 test dataset (max_samples={max_samples})...")
    
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        
        dataset = load_dataset("wikitext_2", "wikitext-2-raw-v1", split="test")
        texts = dataset["text"][:max_samples]
        logger.info(f"Loaded {len(texts)} samples")
        return texts
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return []


def compute_perplexity(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 8
) -> float:
    """Compute perplexity on texts."""
    device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    total_loss = 0.0
    total_tokens = 0
    
    logger.info(f"Computing perplexity on {len(texts)} texts...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for language modeling loss
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            
            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item() * shift_labels.numel()
            total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"Perplexity: {perplexity:.2f}")
    return perplexity


def measure_memory(model: nn.Module) -> Dict:
    """Measure GPU memory usage."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e6  # MB
        memory_reserved = torch.cuda.memory_reserved(0) / 1e6  # MB
        return {
            "allocated_mb": memory_allocated,
            "reserved_mb": memory_reserved,
            "allocated_gb": memory_allocated / 1024,
            "reserved_gb": memory_reserved / 1024
        }
    else:
        return {"allocated_mb": 0, "reserved_mb": 0, "allocated_gb": 0, "reserved_gb": 0}


def measure_inference_speed(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    num_runs: int = 10
) -> Dict:
    """Measure inference speed."""
    device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    times = []
    text = texts[0] if texts else "Hello world"
    
    encoding = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    logger.info(f"Measuring inference speed ({num_runs} runs)...")
    
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "avg_time_sec": avg_time,
        "min_time_sec": min_time,
        "max_time_sec": max_time,
        "samples_per_sec": 1.0 / avg_time if avg_time > 0 else 0
    }


def run_benchmark_evaluation():
    """Run comprehensive benchmark evaluation."""
    logger.info("=" * 60)
    logger.info("Benchmark Evaluation for Quantization Levels")
    logger.info("=" * 60)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen3.5-2B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Load benchmark data
    texts = load_wikitext2_test(max_samples=50)
    if not texts:
        logger.error("Failed to load benchmark texts")
        return
    
    # Benchmark each quantization level
    results = {}
    quantization_levels = get_quantization_levels()
    
    for level in quantization_levels:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking {level.value} quantization")
        logger.info(f"{'='*60}")
        
        # Apply quantization
        quantizer = Quantizer(level)
        quantized_model = apply_quantization_to_model(model, quantizer)
        
        # Measure memory
        memory = measure_memory(quantized_model)
        logger.info(f"Memory: {memory['allocated_gb']:.2f} GB allocated")
        
        # Measure perplexity
        perplexity = compute_perplexity(quantized_model, tokenizer, texts)
        
        # Measure speed
        speed = measure_inference_speed(quantized_model, tokenizer, texts)
        logger.info(f"Inference time: {speed['avg_time_sec']*1000:.2f} ms")
        
        # Store results
        results[level.value] = {
            "quantization_level": level.value,
            "memory_ratio": quantizer.get_memory_ratio(),
            "memory_gb": memory['allocated_gb'],
            "perplexity": perplexity,
            "inference_time_ms": speed['avg_time_sec'] * 1000,
            "samples_per_sec": speed['samples_per_sec']
        }
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    results_path = BENCHMARKS_DIR / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved benchmark results: {results_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Level\t\tMemory\tPerplexity\tTime (ms)")
    logger.info("-" * 60)
    
    for level in ['fp16', 'int8', 'int4', '1bit']:
        if level in results:
            r = results[level]
            logger.info(f"{level}\t\t{r['memory_ratio']:.3f}\t{r['perplexity']:.2f}\t\t{r['inference_time_ms']:.2f}")
    
    logger.info("=" * 60)
    logger.info("✓ Benchmark evaluation completed!")
    
    return results


if __name__ == "__main__":
    run_benchmark_evaluation()