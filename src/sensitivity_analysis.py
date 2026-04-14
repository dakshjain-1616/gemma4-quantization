#!/usr/bin/env python3
"""
Layer-by-layer sensitivity analysis for quantization.
Analyzes which layers tolerate 1-bit quantization and which degrade.
Outputs: sensitivity map CSV/JSON, benchmark results.
"""

import os
import sys
import logging
import json
import csv
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

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
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_layer_sensitivity(
    layer: nn.Linear,
    layer_idx: int,
    layer_name: str
) -> Dict:
    """
    Analyze sensitivity of a single layer to different quantization levels.
    
    Args:
        layer: Linear layer to analyze
        layer_idx: Layer index
        layer_name: Layer name
        
    Returns:
        Dictionary with sensitivity metrics for all quantization levels
    """
    original_weight = layer.weight.data.clone()
    original_norm = original_weight.norm().item()
    
    results = {
        'layer_idx': layer_idx,
        'layer_name': layer_name,
        'weight_shape': list(layer.weight.shape),
        'original_norm': original_norm
    }
    
    # Analyze each quantization level
    for level in get_quantization_levels():
        quantizer = Quantizer(level)
        quantized_weight = quantizer.quantize_weight(original_weight)
        
        # Compute metrics
        absolute_error = (original_weight - quantized_weight).abs().mean().item()
        relative_error = absolute_error / (original_norm + 1e-6)
        correlation = torch.nn.functional.cosine_similarity(
            original_weight.flatten().unsqueeze(0),
            quantized_weight.flatten().unsqueeze(0)
        ).item()
        
        # Sensitivity score: how much the layer degrades (lower = more sensitive)
        # Using correlation as primary metric
        sensitivity_score = correlation
        
        results[level.value] = {
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'correlation': correlation,
            'sensitivity_score': sensitivity_score,
            'memory_ratio': quantizer.get_memory_ratio()
        }
    
    return results


def extract_linear_layers(model: nn.Module) -> List[Tuple[int, str, nn.Linear]]:
    """
    Extract all linear layers from a model with their indices and names.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of (index, name, layer) tuples
    """
    linear_layers = []
    
    for idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Linear):
            linear_layers.append((idx, name, module))
    
    return linear_layers


def perform_sensitivity_analysis(model: nn.Module) -> List[Dict]:
    """
    Perform layer-by-layer sensitivity analysis on all linear layers.
    
    Args:
        model: Model to analyze
        
    Returns:
        List of sensitivity results for each layer
    """
    logger.info("Extracting linear layers...")
    linear_layers = extract_linear_layers(model)
    logger.info(f"Found {len(linear_layers)} linear layers")
    
    results = []
    
    logger.info("Analyzing layer sensitivity...")
    for idx, name, layer in tqdm(linear_layers, desc="Layers"):
        result = analyze_layer_sensitivity(layer, idx, name)
        results.append(result)
    
    return results


def save_sensitivity_map(results: List[Dict], output_format: str = "both"):
    """
    Save sensitivity analysis results to CSV and JSON.
    
    Args:
        results: List of sensitivity results
        output_format: "csv", "json", or "both"
    """
    logger.info("Saving sensitivity map...")
    
    # Save as JSON
    if output_format in ["json", "both"]:
        json_path = RESULTS_DIR / "sensitivity_map.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved JSON: {json_path}")
    
    # Save as CSV
    if output_format in ["csv", "both"]:
        csv_path = RESULTS_DIR / "sensitivity_map.csv"
        
        # Flatten results for CSV
        csv_rows = []
        for result in results:
            row = {
                'layer_idx': result['layer_idx'],
                'layer_name': result['layer_name'],
                'weight_shape': str(result['weight_shape']),
                'original_norm': result['original_norm']
            }
            
            # Add metrics for each quantization level
            for level in ['fp16', 'int8', 'int4', '1bit']:
                if level in result:
                    metrics = result[level]
                    row[f'{level}_absolute_error'] = metrics['absolute_error']
                    row[f'{level}_relative_error'] = metrics['relative_error']
                    row[f'{level}_correlation'] = metrics['correlation']
                    row[f'{level}_sensitivity_score'] = metrics['sensitivity_score']
                    row[f'{level}_memory_ratio'] = metrics['memory_ratio']
            
            csv_rows.append(row)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        
        logger.info(f"Saved CSV: {csv_path}")


def identify_sensitive_layers(results: List[Dict], threshold: float = 0.9) -> Dict:
    """
    Identify layers that are highly sensitive to 1-bit quantization.
    
    Args:
        results: Sensitivity analysis results
        threshold: Correlation threshold for sensitivity (below = sensitive)
        
    Returns:
        Dictionary with sensitive layer information
    """
    sensitive_layers = []
    tolerant_layers = []
    
    for result in results:
        bit1_metrics = result.get('1bit', {})
        correlation = bit1_metrics.get('sensitivity_score', 0)
        
        if correlation < threshold:
            sensitive_layers.append({
                'layer_idx': result['layer_idx'],
                'layer_name': result['layer_name'],
                'correlation': correlation,
                'relative_error': bit1_metrics.get('relative_error', 0)
            })
        else:
            tolerant_layers.append({
                'layer_idx': result['layer_idx'],
                'layer_name': result['layer_name'],
                'correlation': correlation
            })
    
    summary = {
        'total_layers': len(results),
        'sensitive_layers_count': len(sensitive_layers),
        'tolerant_layers_count': len(tolerant_layers),
        'threshold': threshold,
        'sensitive_layers': sensitive_layers,
        'tolerant_layers': tolerant_layers
    }
    
    return summary


def generate_sensitivity_report(results: List[Dict]) -> str:
    """
    Generate a human-readable sensitivity analysis report.
    
    Args:
        results: Sensitivity analysis results
        
    Returns:
        Report text
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("LAYER-BY-LAYER SENSITIVITY ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Summary statistics
    avg_correlations = {}
    for level in ['fp16', 'int8', 'int4', '1bit']:
        correlations = [r[level]['correlation'] for r in results if level in r]
        avg_correlations[level] = sum(correlations) / len(correlations) if correlations else 0
    
    report_lines.append("AVERAGE CORRELATION BY QUANTIZATION LEVEL:")
    for level, avg_corr in avg_correlations.items():
        report_lines.append(f"  {level}: {avg_corr:.4f}")
    report_lines.append("")
    
    # Identify most sensitive layers for 1-bit
    bit1_results = [(r['layer_idx'], r['layer_name'], r['1bit']['correlation']) 
                    for r in results if '1bit' in r]
    bit1_results.sort(key=lambda x: x[2])  # Sort by correlation (lowest first)
    
    report_lines.append("MOST SENSITIVE LAYERS TO 1-BIT QUANTIZATION:")
    report_lines.append("(Lowest correlation = most sensitive)")
    for idx, name, corr in bit1_results[:10]:
        report_lines.append(f"  Layer {idx}: {name} (correlation: {corr:.4f})")
    report_lines.append("")
    
    # Identify most tolerant layers
    report_lines.append("MOST TOLERANT LAYERS TO 1-BIT QUANTIZATION:")
    report_lines.append("(Highest correlation = most tolerant)")
    for idx, name, corr in bit1_results[-10:]:
        report_lines.append(f"  Layer {idx}: {name} (correlation: {corr:.4f})")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS:")
    sensitive_count = sum(1 for r in results if r.get('1bit', {}).get('correlation', 1) < 0.9)
    tolerant_count = len(results) - sensitive_count
    
    report_lines.append(f"  - {sensitive_count}/{len(results)} layers are sensitive to 1-bit quantization (correlation < 0.9)")
    report_lines.append(f"  - {tolerant_count}/{len(results)} layers tolerate 1-bit quantization well")
    
    if sensitive_count > len(results) * 0.5:
        report_lines.append("  - WARNING: Majority of layers are sensitive to 1-bit")
        report_lines.append("  - Recommendation: Use INT4 or INT8 for better quality")
    else:
        report_lines.append("  - 1-bit quantization may be feasible for this model")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


def main():
    """Main entry point for sensitivity analysis"""
    logger.info("=" * 60)
    logger.info("Layer-by-Layer Sensitivity Analysis")
    logger.info("=" * 60)
    
    logger.info("Loading model for analysis...")
    
    try:
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-4-E2B-it",
            torch_dtype=torch.float32,  # Use FP32 for analysis
            device_map="auto"
        )
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.warning(f"Failed to load model: {e}")
        logger.info("Using synthetic model for demonstration...")
        
        # Create synthetic model with multiple linear layers
        class SyntheticModel(nn.Module):
            def __init__(self, num_layers=20):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(512, 512) for _ in range(num_layers)
                ])
                self.output = nn.Linear(512, 1000)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.output(x)
        
        model = SyntheticModel(num_layers=20)
        logger.info(f"Created synthetic model with {len(model.layers)} layers")
    
    # Perform sensitivity analysis
    results = perform_sensitivity_analysis(model)
    
    # Save results
    save_sensitivity_map(results, output_format="both")
    
    # Identify sensitive layers
    summary = identify_sensitive_layers(results, threshold=0.9)
    
    # Save summary
    summary_path = RESULTS_DIR / "sensitivity_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")
    
    # Generate and save report
    report = generate_sensitivity_report(results)
    print(report)
    
    report_path = RESULTS_DIR / "sensitivity_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved report: {report_path}")
    
    logger.info("\n✓ Sensitivity analysis completed!")
    logger.info(f"  - Total layers analyzed: {len(results)}")
    logger.info(f"  - Sensitive layers (1-bit): {summary['sensitive_layers_count']}")
    logger.info(f"  - Tolerant layers (1-bit): {summary['tolerant_layers_count']}")


if __name__ == "__main__":
    main()