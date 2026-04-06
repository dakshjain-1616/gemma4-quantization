# Quantization Feasibility Study — google/gemma-4-E2B-it
Generated: 2026-04-06 06:58:39 UTC

## Perplexity Results
| Precision | Perplexity | vs FP16 |
|-----------|-----------|---------|
| FP16 | 127575.28 | baseline |
| INT8 | 87204.77 | 0.7× |
| INT4 | 3594112225964632.00 | 28172481499.8× |
| BIT1 | 65327058492.17 | 512066.7× |

## Layer Sensitivity (1-bit)
- Total layers: 566
- Sensitive (cos<0.90): 418 (74%)
- Cosine similarity range: 0.661 – 0.967