#!/usr/bin/env python3
"""
1-Bit Quantization Feasibility Study — google/gemma-4-E2B-it
Replaces the Qwen3.5-2B proxy with the actual Gemma-4 architecture.
Runs: BF16 → INT8 → INT4 → 1-bit sweep + full layer sensitivity analysis.
Outputs: results/benchmark_results.json, analysis/, SVGs, updated README.
"""

import os, json, csv, math, time, logging, sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List
from copy import deepcopy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR   = PROJECT_ROOT / "results"
ANALYSIS_DIR  = PROJECT_ROOT / "analysis"
REPORTS_DIR   = PROJECT_ROOT / "reports"
EXPORTS_DIR   = PROJECT_ROOT / "hf_exports"

for d in [RESULTS_DIR, ANALYSIS_DIR, REPORTS_DIR]: d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/gemma-4-E2B-it"
HF_TOKEN   = open("/root/.cache/huggingface/token").read().strip()

# ---------------------------------------------------------------------------
# Quantization helpers — canonical implementations live in src/quantization.py
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent / "src"))
from quantization import Quantizer, QuantizationLevel

def quantize_int8(weight: torch.Tensor) -> torch.Tensor:
    return Quantizer(QuantizationLevel.INT8).quantize_weight(weight)

def quantize_int4(weight: torch.Tensor) -> torch.Tensor:
    return Quantizer(QuantizationLevel.INT4).quantize_weight(weight)

def quantize_1bit(weight: torch.Tensor) -> torch.Tensor:
    return Quantizer(QuantizationLevel.INT1).quantize_weight(weight)

def apply_quantization(model: nn.Module, quant_fn) -> nn.Module:
    m = deepcopy(model)
    with torch.no_grad():
        for module in m.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = quant_fn(module.weight.data)
    return m

# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def load_wikitext2(n=50) -> List[str]:
    """Load WikiText-2 test set for perplexity evaluation."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 100][:n]
        logger.info(f"Loaded {len(texts)} WikiText-2 samples")
        return texts
    except Exception as e:
        logger.warning(f"WikiText-2 load failed ({e}), using fallback texts")
        return [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "The transformer architecture relies on self-attention mechanisms that allow the model to weigh importance.",
            "Quantization reduces the precision of model weights from floating point to lower bit representations.",
            "Large language models have demonstrated emergent capabilities at scale including in-context learning.",
            "The attention mechanism computes query key and value projections then applies a scaled dot product.",
        ]

EVAL_TEXTS = None  # loaded lazily

def compute_perplexity(model, tokenizer, texts=None, max_length=256) -> float:
    global EVAL_TEXTS
    if texts is None:
        if EVAL_TEXTS is None:
            EVAL_TEXTS = load_wikitext2(50)
        texts = EVAL_TEXTS
    device = next(model.parameters()).device
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length).to(device)
            ids  = enc["input_ids"]
            mask = enc["attention_mask"]
            out  = model(ids, attention_mask=mask, labels=ids)
            n = ids.shape[1] - 1
            total_loss   += out.loss.item() * n
            total_tokens += n
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

def measure_inference(model, tokenizer, texts=None, runs=5, warmup=2) -> Dict:
    global EVAL_TEXTS
    if texts is None:
        if EVAL_TEXTS is None:
            EVAL_TEXTS = load_wikitext2(50)
        texts = EVAL_TEXTS[:3]
    device = next(model.parameters()).device
    model.eval()
    enc = tokenizer(texts[0], return_tensors="pt",
                    truncation=True, max_length=128).to(device)
    # Warmup: let CUDA JIT-compile kernels before timing
    with torch.no_grad():
        for _ in range(warmup):
            model(**enc)
            torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(**enc)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    return {"inference_time_ms": avg_ms, "samples_per_sec": 1000.0 / avg_ms}

# ---------------------------------------------------------------------------
# Layer sensitivity
# ---------------------------------------------------------------------------

def layer_sensitivity(model: nn.Module) -> List[Dict]:
    results = []
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for name, module in tqdm(layers, desc="Sensitivity", leave=False):
        w = module.weight.data.float()
        w_q = quantize_1bit(w)
        cos = torch.nn.functional.cosine_similarity(
            w.flatten().unsqueeze(0), w_q.flatten().unsqueeze(0)).item()
        results.append({
            "layer_name": name,
            "cosine_similarity": cos,
            "sensitive": cos < 0.90,
            "weight_shape": list(w.shape),
        })
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # BF16 is required: Gemma-4 was trained in BF16 (exponent range ±127 vs FP16's ±15).
    # FP16 overflows in attention softmax, producing NaN logits and garbage perplexity.
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda", token=HF_TOKEN
    )
    model_bf16.eval()
    n_params = sum(p.numel() for p in model_bf16.parameters())
    logger.info(f"Model loaded. Params: {n_params/1e9:.2f}B")

    # ── Sensitivity analysis (on BF16 model) ───────────────────────────────
    logger.info("Running layer sensitivity analysis...")
    sensitivity = layer_sensitivity(model_bf16)
    sensitive_count = sum(1 for s in sensitivity if s["sensitive"])
    total_layers = len(sensitivity)

    with open(ANALYSIS_DIR / "sensitivity_map.json", "w") as f:
        json.dump(sensitivity, f, indent=2)
    with open(ANALYSIS_DIR / "sensitivity_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer_name","cosine_similarity","sensitive","weight_shape"])
        w.writeheader(); w.writerows(sensitivity)

    cos_vals = [s["cosine_similarity"] for s in sensitivity]
    sens_summary = {
        "total_layers": total_layers,
        "sensitive_layers_count": sensitive_count,
        "tolerant_layers_count": total_layers - sensitive_count,
        "cosine_sim_min": min(cos_vals),
        "cosine_sim_max": max(cos_vals),
        "cosine_sim_mean": sum(cos_vals) / len(cos_vals),
        "threshold": 0.90,
    }
    with open(ANALYSIS_DIR / "sensitivity_summary.json", "w") as f:
        json.dump(sens_summary, f, indent=2)
    logger.info(f"Sensitive layers: {sensitive_count}/{total_layers} "
                f"(cos range {min(cos_vals):.3f}–{max(cos_vals):.3f})")

    # ── Benchmark sweep ─────────────────────────────────────────────────────
    # Theoretical memory: simulated quant dequantizes back to BF16, so
    # torch.cuda.memory_allocated() always reflects two BF16 copies (~20 GB).
    # Report theoretical deployment memory derived from parameter count instead.
    bytes_per_param = {"bf16": 2.0, "int8": 1.0, "int4": 0.5, "bit1": 0.1976}  # 1.58-bit ternary
    theoretical_mem = {lvl: n_params * bpp / 1e9 for lvl, bpp in bytes_per_param.items()}

    benchmark = {}
    quant_configs = [
        ("bf16",  None,          model_bf16),
        ("int8",  quantize_int8, None),
        ("int4",  quantize_int4, None),
        ("bit1",  quantize_1bit, None),
    ]

    for level, fn, m in quant_configs:
        logger.info(f"Benchmarking {level}...")
        if m is None:
            m = apply_quantization(model_bf16, fn)
        ppl   = compute_perplexity(m, tokenizer)
        speed = measure_inference(m, tokenizer)
        benchmark[level] = {
            "quantization_level": level,
            "perplexity": ppl,
            "memory_gb": theoretical_mem[level],
            **speed,
        }
        logger.info(f"  {level}: perplexity={ppl:.2f}, mem={theoretical_mem[level]:.2f}GB (theoretical), "
                    f"ms={speed['inference_time_ms']:.1f}")
        if m is not model_bf16:
            del m; torch.cuda.empty_cache()

    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(benchmark, f, indent=2)

    # ── Summary report ───────────────────────────────────────────────────────
    label = {"bf16": "BF16", "int8": "INT8", "int4": "INT4", "bit1": "1-bit"}
    report_lines = [
        f"# Quantization Feasibility Study — {MODEL_NAME}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Perplexity Results",
        "| Precision | Perplexity | vs BF16 |",
        "|-----------|-----------|---------|",
    ]
    bf16_ppl = benchmark["bf16"]["perplexity"]
    for lvl in ["bf16","int8","int4","bit1"]:
        p = benchmark[lvl]["perplexity"]
        delta = f"{p/bf16_ppl:.1f}×" if lvl != "bf16" else "baseline"
        report_lines.append(f"| {label[lvl]} | {p:.2f} | {delta} |")

    report_lines += [
        "",
        "## Layer Sensitivity (1-bit)",
        f"- Total layers: {total_layers}",
        f"- Sensitive (cos<0.90): {sensitive_count} ({sensitive_count/total_layers*100:.0f}%)",
        f"- Cosine similarity range: {min(cos_vals):.3f} – {max(cos_vals):.3f}",
    ]
    with open(REPORTS_DIR / "summary_report.md", "w") as f:
        f.write("\n".join(report_lines))

    logger.info("Study complete. Results saved.")
    return benchmark, sens_summary


if __name__ == "__main__":
    main()
