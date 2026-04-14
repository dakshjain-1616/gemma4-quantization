# Quantization Feasibility Study — google/gemma-4-E2B-it
Generated: 2026-04-14 12:16:11 UTC

## Perplexity Results
| Precision | Perplexity | vs BF16 |
|-----------|-----------|---------|
| BF16 | 128453.65 | baseline |
| INT8 | 92305.84 | 0.7× |
| INT4 | 3307064179911382.00 | 25745195140.8× |
| 1-bit | 51310066740.89 | 399444.2× |

## Layer Sensitivity (1-bit)
- Total layers: 566
- Sensitive (cos<0.90): 418 (74%)
- Cosine similarity range: 0.661 – 0.967
