# Corrected Qwen VL Benchmark

Generated 2026-10-29 09:43:35 +0100 with 100 fixed examples per task.
All calls are serial, single image, single request, and at most 256 output tokens (2,048 for Thinking).

## Task scores and deltas

| Model | Runner | Reference | AI2D | Δ AI2D | TextVQA | Δ TextVQA | Status |
|---|---|---|---:|---:|---:|---:|---|
| Qwen/Qwen2.5-VL-7B-Instruct | Transformers (model card) | — | 83.0% | — | 85.2% | — | complete |
| Qwen/Qwen3-VL-2B-Instruct | Transformers (model card) | — | 72.0% | — | 77.3% | — | complete |
| Qwen/Qwen3-VL-2B-Thinking | Transformers (model card) | — | 70.0% | — | 68.0% | — | complete |
| Qwen/Qwen3-VL-8B-Instruct | Transformers (model card) | — | 84.0% | — | 81.2% | — | complete |
| Qwen/Qwen3-VL-8B-Thinking | Transformers (model card) | — | 80.0% | — | 72.1% | — | complete |
| zmli/G-Substrate-Qwen3-VL-2B | Transformers (model card) | Qwen/Qwen3-VL-2B-Instruct | 58.0% | -14.0 pp | 41.0% | -36.3 pp | complete |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-2B-Instruct | 78.0% | +6.0 pp | 78.1% | +0.8 pp | complete |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q8_K_XL | llama.cpp (UD-Q8_K_XL) | Qwen/Qwen3-VL-2B-Instruct | 79.0% | +7.0 pp | 76.8% | -0.5 pp | complete |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-2B-Thinking | 64.0% | -6.0 pp | 69.5% | +1.5 pp | complete |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q8_K_XL | llama.cpp (UD-Q8_K_XL) | Qwen/Qwen3-VL-2B-Thinking | 62.0% | -8.0 pp | 66.8% | -1.2 pp | complete |
| unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | — | 90.0% | — | 78.8% | — | complete |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL | llama.cpp (UD-Q8_K_XL) | Qwen/Qwen3-VL-8B-Instruct | 85.0% | +1.0 pp | 81.1% | -0.1 pp | complete |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-8B-Thinking | 80.0% | +0.0 pp | 72.8% | +0.7 pp | complete |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q8_K_XL | llama.cpp (UD-Q8_K_XL) | Qwen/Qwen3-VL-8B-Thinking | 81.0% | +1.0 pp | 75.3% | +3.2 pp | complete |
| Qwen/Qwen2-VL-2B-Instruct | Transformers (model card) | — | 67.0% | — | 80.0% | — | complete |
| Qwen/Qwen2-VL-7B-Instruct | Transformers (model card) | — | 80.0% | — | 88.0% | — | complete |
| Qwen/Qwen2.5-VL-3B-Instruct | Transformers (model card) | — | 79.0% | — | 71.9% | — | complete |
| Qwen/Qwen3-VL-4B-Instruct | Transformers (model card) | — | 78.0% | — | 80.2% | — | complete |
| Qwen/Qwen3-VL-4B-Thinking | Transformers (model card) | — | 77.0% | — | 72.2% | — | complete |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-2B-Instruct | 67.0% | -5.0 pp | 76.1% | -1.2 pp | complete |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q8_K_XL | llama.cpp (UD-Q8_K_XL) | Qwen/Qwen3-VL-2B-Instruct | 71.0% | -1.0 pp | 78.1% | +0.8 pp | complete |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-2B-Thinking | 50.0% | -20.0 pp | 67.1% | -0.9 pp | complete |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q8_K_XL | llama.cpp (UD-Q8_K_XL) | Qwen/Qwen3-VL-2B-Thinking | 53.0% | -17.0 pp | 68.6% | +0.6 pp | complete |
| unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | — | 81.0% | — | 82.4% | — | complete |
| unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | — | 85.0% | — | 84.0% | — | complete |
| unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | — | 85.0% | — | 73.0% | — | complete |
| unsloth/Qwen3-VL-32B-Instruct-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | — | 89.0% | — | 75.4% | — | complete |
| unsloth/Qwen3-VL-32B-Thinking-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | — | 88.0% | — | 74.8% | — | complete |
| unsloth/Qwen3-VL-4B-Instruct-1M-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-4B-Instruct | 79.0% | +1.0 pp | 78.8% | -1.4 pp | complete |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-4B-Instruct | 78.0% | +0.0 pp | 77.9% | -2.3 pp | complete |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL | llama.cpp (UD-Q8_K_XL) | Qwen/Qwen3-VL-4B-Instruct | 79.0% | +1.0 pp | 79.5% | -0.7 pp | complete |
| unsloth/Qwen3-VL-4B-Thinking-1M-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-4B-Thinking | 70.0% | -7.0 pp | 72.6% | +0.4 pp | complete |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-4B-Thinking | 70.0% | -7.0 pp | 75.3% | +3.1 pp | complete |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q8_K_XL | llama.cpp (UD-Q8_K_XL) | Qwen/Qwen3-VL-4B-Thinking | 76.0% | -1.0 pp | 73.8% | +1.6 pp | complete |
| unsloth/Qwen3-VL-8B-Instruct-1M-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-8B-Instruct | 81.0% | -3.0 pp | 83.1% | +1.9 pp | complete |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-8B-Instruct | 81.0% | -3.0 pp | 80.0% | -1.2 pp | complete |
| unsloth/Qwen3-VL-8B-Thinking-1M-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | Qwen/Qwen3-VL-8B-Thinking | 77.0% | -3.0 pp | 76.4% | +4.3 pp | complete |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | — | 82.0% | — | 72.3% | — | complete |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q5_K_XL | llama.cpp (UD-Q5_K_XL) | — | 85.0% | — | 72.3% | — | complete |
| unsloth/Qwen2.5-VL-72B-Instruct-GGUF:UD-IQ2_XXS | llama.cpp (UD-IQ2_XXS) | — | 82.0% | — | 73.5% | — | complete |
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | llama.cpp (IQ2_XXS) | Qwen/Qwen2-VL-72B-Instruct | 84.0% | — | 87.1% | — | complete |
| unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit | Transformers (Unsloth BNB 4-bit model card) | Qwen/Qwen3-VL-8B-Thinking | 76.0% | -4.0 pp | 71.9% | -0.2 pp | complete |
| unsloth/Qwen3-VL-8B-Thinking-bnb-4bit | Transformers (Unsloth BNB 4-bit model card) | Qwen/Qwen3-VL-8B-Thinking | 79.0% | -1.0 pp | 73.4% | +1.3 pp | complete |

## Per task VRAM and speed

| Model | Task | Completed / parse errors | Peak GPU used | Mean seconds/example | Effective output tokens/second | Context | KV cache |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen/Qwen2.5-VL-7B-Instruct | ai2d | 100/100; 0 | 17.94 GiB | 0.17 | 11.5 | 125K | Transformers HQQ int8 |
| Qwen/Qwen2.5-VL-7B-Instruct | textvqa | 100/100; 0 | 17.94 GiB | 0.26 | 16.4 | 125K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-2B-Instruct | ai2d | 100/100; 0 | 6.06 GiB | 0.15 | 31.8 | 256K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-2B-Instruct | textvqa | 100/100; 0 | 6.06 GiB | 0.16 | 30.3 | 256K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-2B-Thinking | ai2d | 100/100; 8 | 6.28 GiB | 13.22 | 73.3 | 256K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-2B-Thinking | textvqa | 100/100; 0 | 6.28 GiB | 2.93 | 69.0 | 256K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-8B-Instruct | ai2d | 100/100; 0 | 18.65 GiB | 0.18 | 11.6 | 256K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-8B-Instruct | textvqa | 100/100; 0 | 18.65 GiB | 0.25 | 16.9 | 256K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-8B-Thinking | ai2d | 100/100; 2 | 18.86 GiB | 8.61 | 55.8 | 256K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-8B-Thinking | textvqa | 100/100; 0 | 18.86 GiB | 2.93 | 55.5 | 256K | Transformers HQQ int8 |
| zmli/G-Substrate-Qwen3-VL-2B | ai2d | 100/100; 9 | 10.29 GiB | 0.20 | 25.9 | 256K | Transformers HQQ int8 |
| zmli/G-Substrate-Qwen3-VL-2B | textvqa | 100/100; 0 | 10.29 GiB | 0.70 | 57.5 | 256K | Transformers HQQ int8 |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q4_K_XL | ai2d | 100/100; 0 | 23.60 GiB | 0.27 | 16.4 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 23.60 GiB | 0.24 | 18.7 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q8_K_XL | ai2d | 100/100; 0 | 24.68 GiB | 0.34 | 14.3 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q8_K_XL | textvqa | 100/100; 0 | 24.68 GiB | 0.27 | 17.1 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q4_K_XL | ai2d | 100/100; 24 | 23.71 GiB | 3.52 | 239.8 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q4_K_XL | textvqa | 100/100; 1 | 23.71 GiB | 1.70 | 123.3 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q8_K_XL | ai2d | 100/100; 27 | 25.43 GiB | 5.35 | 169.9 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q8_K_XL | textvqa | 100/100; 4 | 25.43 GiB | 0.93 | 273.4 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL | ai2d | 100/100; 1 | 29.86 GiB | 1.96 | 128.1 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 29.86 GiB | 1.59 | 102.5 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL | ai2d | 100/100; 0 | 23.07 GiB | 0.43 | 4.8 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL | textvqa | 100/100; 0 | 23.07 GiB | 0.47 | 9.2 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q4_K_XL | ai2d | 100/100; 11 | 28.74 GiB | 2.80 | 172.9 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q4_K_XL | textvqa | 100/100; 2 | 28.74 GiB | 1.16 | 128.8 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q8_K_XL | ai2d | 100/100; 6 | 23.04 GiB | 3.36 | 113.0 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q8_K_XL | textvqa | 100/100; 1 | 23.04 GiB | 1.71 | 96.7 | 128K | q8_0 K and V |
| Qwen/Qwen2-VL-2B-Instruct | ai2d | 100/100; 0 | 6.99 GiB | 0.12 | 21.4 | 32K | Transformers HQQ int8 |
| Qwen/Qwen2-VL-2B-Instruct | textvqa | 100/100; 0 | 6.99 GiB | 0.18 | 23.8 | 32K | Transformers HQQ int8 |
| Qwen/Qwen2-VL-7B-Instruct | ai2d | 100/100; 0 | 18.21 GiB | 0.16 | 12.4 | 32K | Transformers HQQ int8 |
| Qwen/Qwen2-VL-7B-Instruct | textvqa | 100/100; 0 | 18.21 GiB | 0.25 | 16.6 | 32K | Transformers HQQ int8 |
| Qwen/Qwen2.5-VL-3B-Instruct | ai2d | 100/100; 0 | 9.36 GiB | 0.15 | 13.5 | 125K | Transformers HQQ int8 |
| Qwen/Qwen2.5-VL-3B-Instruct | textvqa | 100/100; 0 | 9.36 GiB | 0.24 | 20.6 | 125K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-4B-Instruct | ai2d | 100/100; 0 | 10.46 GiB | 0.14 | 15.1 | 256K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-4B-Instruct | textvqa | 100/100; 0 | 10.46 GiB | 0.19 | 23.5 | 256K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-4B-Thinking | ai2d | 100/100; 4 | 10.46 GiB | 10.70 | 60.1 | 256K | Transformers HQQ int8 |
| Qwen/Qwen3-VL-4B-Thinking | textvqa | 100/100; 0 | 10.46 GiB | 2.81 | 58.9 | 256K | Transformers HQQ int8 |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q4_K_XL | ai2d | 100/100; 0 | 20.64 GiB | 0.08 | 58.8 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 20.64 GiB | 0.15 | 31.5 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q8_K_XL | ai2d | 100/100; 0 | 21.79 GiB | 0.09 | 58.6 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q8_K_XL | textvqa | 100/100; 0 | 21.80 GiB | 0.16 | 30.9 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q4_K_XL | ai2d | 100/100; 43 | 20.65 GiB | 2.47 | 467.4 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q4_K_XL | textvqa | 100/100; 6 | 20.65 GiB | 0.72 | 385.4 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q8_K_XL | ai2d | 100/100; 36 | 21.82 GiB | 3.20 | 346.5 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q8_K_XL | textvqa | 100/100; 2 | 21.82 GiB | 0.81 | 292.7 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF:UD-Q4_K_XL | ai2d | 100/100; 0 | 26.90 GiB | 0.15 | 25.5 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 26.90 GiB | 0.27 | 16.4 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL | ai2d | 100/100; 0 | 26.90 GiB | 0.15 | 22.9 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 26.90 GiB | 0.27 | 16.5 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL | ai2d | 100/100; 4 | 26.93 GiB | 1.35 | 251.5 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL | textvqa | 100/100; 1 | 26.93 GiB | 0.91 | 201.2 | 128K | q8_0 K and V |
| unsloth/Qwen3-VL-32B-Instruct-GGUF:UD-Q4_K_XL | ai2d | 100/100; 0 | 31.16 GiB | 0.28 | 7.2 | 64K | q8_0 K and V |
| unsloth/Qwen3-VL-32B-Instruct-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 31.16 GiB | 0.53 | 9.2 | 64K | q8_0 K and V |
| unsloth/Qwen3-VL-32B-Thinking-GGUF:UD-Q4_K_XL | ai2d | 100/100; 5 | 31.15 GiB | 6.87 | 59.9 | 64K | q8_0 K and V |
| unsloth/Qwen3-VL-32B-Thinking-GGUF:UD-Q4_K_XL | textvqa | 100/100; 2 | 31.15 GiB | 2.79 | 51.6 | 64K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Instruct-1M-GGUF:UD-Q4_K_XL | ai2d | 100/100; 0 | 26.26 GiB | 0.10 | 27.4 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Instruct-1M-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 26.26 GiB | 0.19 | 23.3 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q4_K_XL | ai2d | 100/100; 0 | 26.26 GiB | 0.10 | 23.1 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 26.26 GiB | 0.19 | 23.7 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL | ai2d | 100/100; 0 | 28.64 GiB | 0.10 | 22.7 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL | textvqa | 100/100; 0 | 28.65 GiB | 0.20 | 21.9 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Thinking-1M-GGUF:UD-Q4_K_XL | ai2d | 100/100; 24 | 26.27 GiB | 2.72 | 279.5 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Thinking-1M-GGUF:UD-Q4_K_XL | textvqa | 100/100; 2 | 26.27 GiB | 0.84 | 232.9 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q4_K_XL | ai2d | 100/100; 12 | 26.27 GiB | 2.33 | 277.5 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 26.27 GiB | 0.68 | 217.9 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q8_K_XL | ai2d | 100/100; 12 | 28.64 GiB | 3.25 | 196.6 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q8_K_XL | textvqa | 100/100; 0 | 28.65 GiB | 0.95 | 165.5 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Instruct-1M-GGUF:UD-Q4_K_XL | ai2d | 100/100; 0 | 28.77 GiB | 0.14 | 15.2 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Instruct-1M-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 28.77 GiB | 0.29 | 14.8 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL | ai2d | 100/100; 0 | 28.78 GiB | 0.14 | 14.6 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 28.78 GiB | 0.29 | 14.9 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Thinking-1M-GGUF:UD-Q4_K_XL | ai2d | 100/100; 9 | 28.77 GiB | 2.72 | 194.5 | 256K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Thinking-1M-GGUF:UD-Q4_K_XL | textvqa | 100/100; 2 | 28.78 GiB | 1.10 | 155.2 | 256K | q8_0 K and V |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q4_K_XL | ai2d | 100/100; 0 | 31.11 GiB | 0.32 | 7.9 | 64K | q8_0 K and V |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q4_K_XL | textvqa | 100/100; 0 | 31.11 GiB | 0.60 | 8.4 | 64K | q8_0 K and V |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q5_K_XL | ai2d | 100/100; 0 | 29.53 GiB | 0.33 | 7.0 | 32K | q8_0 K and V |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q5_K_XL | textvqa | 100/100; 0 | 29.53 GiB | 0.64 | 7.9 | 32K | q8_0 K and V |
| unsloth/Qwen2.5-VL-72B-Instruct-GGUF:UD-IQ2_XXS | ai2d | 100/100; 0 | 30.19 GiB | 0.47 | 4.2 | 16K | q8_0 K and V |
| unsloth/Qwen2.5-VL-72B-Instruct-GGUF:UD-IQ2_XXS | textvqa | 100/100; 0 | 30.19 GiB | 0.94 | 5.1 | 16K | q8_0 K and V |
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | ai2d | 100/100; 0 | 29.66 GiB | 0.44 | 4.6 | 16K | q8_0 K and V |
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | textvqa | 100/100; 0 | 29.66 GiB | 0.87 | 4.7 | 16K | q8_0 K and V |
| unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit | ai2d | 100/100; 5 | 10.21 GiB | 12.55 | 42.6 | 256K | Transformers HQQ int8 |
| unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit | textvqa | 100/100; 0 | 13.08 GiB | 5.31 | 31.8 | 256K | Transformers HQQ int8 |
| unsloth/Qwen3-VL-8B-Thinking-bnb-4bit | ai2d | 100/100; 2 | 9.35 GiB | 9.58 | 43.4 | 256K | Transformers HQQ int8 |
| unsloth/Qwen3-VL-8B-Thinking-bnb-4bit | textvqa | 100/100; 0 | 9.36 GiB | 3.31 | 42.3 | 256K | Transformers HQQ int8 |

## Edge candidates: estimated total VRAM below 12 GiB

| Model | Peak GPU in benchmark | Est. total VRAM @8K | Est. max context at limit | AI2D | TextVQA | Benchmark context | Mean s/example (AI2D / TextVQA) | Output tokens/s (AI2D / TextVQA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unsloth/Qwen3-VL-8B-Instruct-1M-GGUF:UD-Q4_K_XL | 28.77 GiB | 11.18 GiB | 18K | 81.0% | 83.1% | 256K | 0.14 / 0.29 | 15.2 / 14.8 |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL | 28.78 GiB | 11.19 GiB | 18K | 81.0% | 80.0% | 256K | 0.14 / 0.29 | 14.6 / 14.9 |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL | 28.65 GiB | 11.06 GiB | 20K | 79.0% | 79.5% | 256K | 0.10 / 0.20 | 22.7 / 21.9 |
| Qwen/Qwen3-VL-4B-Instruct | 10.46 GiB | 10.83 GiB | 22K | 78.0% | 80.2% | 256K | 0.14 / 0.19 | 15.1 / 23.5 |
| unsloth/Qwen3-VL-4B-Instruct-1M-GGUF:UD-Q4_K_XL | 26.26 GiB | 8.67 GiB | 52K | 79.0% | 78.8% | 256K | 0.10 / 0.19 | 27.4 / 23.3 |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q4_K_XL | 23.60 GiB | 10.13 GiB | 40K | 78.0% | 78.1% | 256K | 0.27 / 0.24 | 16.4 / 18.7 |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q4_K_XL | 26.26 GiB | 8.67 GiB | 52K | 78.0% | 77.9% | 256K | 0.10 / 0.19 | 23.1 / 23.7 |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q8_K_XL | 24.68 GiB | 11.20 GiB | 21K | 79.0% | 76.8% | 256K | 0.34 / 0.27 | 14.3 / 17.1 |
| unsloth/Qwen3-VL-8B-Thinking-1M-GGUF:UD-Q4_K_XL | 28.78 GiB | 11.19 GiB | 18K | 77.0% | 76.4% | 256K | 2.72 / 1.10 | 194.5 / 155.2 |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q4_K_XL | 28.74 GiB | 11.15 GiB | 19K | 80.0% | 72.8% | 256K | 2.80 / 1.16 | 172.9 / 128.8 |
| unsloth/Qwen3-VL-8B-Thinking-bnb-4bit | 9.36 GiB | 9.23 GiB | 42K | 79.0% | 73.4% | 256K | 9.58 / 3.31 | 43.4 / 42.3 |
| Qwen/Qwen2.5-VL-3B-Instruct | 9.36 GiB | 9.06 GiB | 125K | 79.0% | 71.9% | 125K | 0.15 / 0.24 | 13.5 / 20.6 |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q8_K_XL | 28.65 GiB | 11.06 GiB | 20K | 76.0% | 73.8% | 256K | 3.25 / 0.95 | 196.6 / 165.5 |
| Qwen/Qwen3-VL-2B-Instruct | 6.06 GiB | 6.42 GiB | 98K | 72.0% | 77.3% | 256K | 0.15 / 0.16 | 31.8 / 30.3 |
| Qwen/Qwen3-VL-4B-Thinking | 10.46 GiB | 10.83 GiB | 22K | 77.0% | 72.2% | 256K | 10.70 / 2.81 | 60.1 / 58.9 |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q8_K_XL | 21.80 GiB | 8.32 GiB | 71K | 71.0% | 78.1% | 256K | 0.09 / 0.16 | 58.6 / 30.9 |
| unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit | 13.08 GiB | 10.56 GiB | 26K | 76.0% | 71.9% | 256K | 12.55 / 5.31 | 42.6 / 31.8 |
| Qwen/Qwen2-VL-2B-Instruct | 6.99 GiB | 6.12 GiB | 32K | 67.0% | 80.0% | 32K | 0.12 / 0.18 | 21.4 / 23.8 |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q4_K_XL | 26.27 GiB | 8.68 GiB | 52K | 70.0% | 75.3% | 256K | 2.33 / 0.68 | 277.5 / 217.9 |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q4_K_XL | 20.64 GiB | 7.16 GiB | 91K | 67.0% | 76.1% | 256K | 0.08 / 0.15 | 58.8 / 31.5 |
| unsloth/Qwen3-VL-4B-Thinking-1M-GGUF:UD-Q4_K_XL | 26.27 GiB | 8.68 GiB | 52K | 70.0% | 72.6% | 256K | 2.72 / 0.84 | 279.5 / 232.9 |
| Qwen/Qwen3-VL-2B-Thinking | 6.28 GiB | 6.45 GiB | 98K | 70.0% | 68.0% | 256K | 13.22 / 2.93 | 73.3 / 69.0 |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q4_K_XL | 23.71 GiB | 10.23 GiB | 38K | 64.0% | 69.5% | 256K | 3.52 / 1.70 | 239.8 / 123.3 |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q8_K_XL | 25.43 GiB | 11.95 GiB | 8K | 62.0% | 66.8% | 256K | 5.35 / 0.93 | 169.9 / 273.4 |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q8_K_XL | 21.82 GiB | 8.35 GiB | 70K | 53.0% | 68.6% | 256K | 3.20 / 0.81 | 346.5 / 292.7 |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q4_K_XL | 20.65 GiB | 7.18 GiB | 90K | 50.0% | 67.1% | 256K | 2.47 / 0.72 | 467.4 / 385.4 |
| zmli/G-Substrate-Qwen3-VL-2B | 10.29 GiB | 10.42 GiB | 33K | 58.0% | 41.0% | 256K | 0.20 / 0.70 | 25.9 / 57.5 |

## Edge candidates: estimated total VRAM below 16 GiB

| Model | Peak GPU in benchmark | Est. total VRAM @8K | Est. max context at limit | AI2D | TextVQA | Benchmark context | Mean s/example (AI2D / TextVQA) | Output tokens/s (AI2D / TextVQA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL | 23.07 GiB | 15.04 GiB | 20K | 85.0% | 81.1% | 128K | 0.43 / 0.47 | 4.8 / 9.2 |
| unsloth/Qwen3-VL-8B-Instruct-1M-GGUF:UD-Q4_K_XL | 28.77 GiB | 11.18 GiB | 72K | 81.0% | 83.1% | 256K | 0.14 / 0.29 | 15.2 / 14.8 |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL | 28.78 GiB | 11.19 GiB | 72K | 81.0% | 80.0% | 256K | 0.14 / 0.29 | 14.6 / 14.9 |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL | 28.65 GiB | 11.06 GiB | 74K | 79.0% | 79.5% | 256K | 0.10 / 0.20 | 22.7 / 21.9 |
| Qwen/Qwen3-VL-4B-Instruct | 10.46 GiB | 10.83 GiB | 73K | 78.0% | 80.2% | 256K | 0.14 / 0.19 | 15.1 / 23.5 |
| unsloth/Qwen3-VL-4B-Instruct-1M-GGUF:UD-Q4_K_XL | 26.26 GiB | 8.67 GiB | 106K | 79.0% | 78.8% | 256K | 0.10 / 0.19 | 27.4 / 23.3 |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q8_K_XL | 23.04 GiB | 15.01 GiB | 21K | 81.0% | 75.3% | 128K | 3.36 / 1.71 | 113.0 / 96.7 |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q4_K_XL | 23.60 GiB | 10.13 GiB | 108K | 78.0% | 78.1% | 256K | 0.27 / 0.24 | 16.4 / 18.7 |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q4_K_XL | 26.26 GiB | 8.67 GiB | 106K | 78.0% | 77.9% | 256K | 0.10 / 0.19 | 23.1 / 23.7 |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q8_K_XL | 24.68 GiB | 11.20 GiB | 90K | 79.0% | 76.8% | 256K | 0.34 / 0.27 | 14.3 / 17.1 |
| unsloth/Qwen3-VL-8B-Thinking-1M-GGUF:UD-Q4_K_XL | 28.78 GiB | 11.19 GiB | 72K | 77.0% | 76.4% | 256K | 2.72 / 1.10 | 194.5 / 155.2 |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q4_K_XL | 28.74 GiB | 11.15 GiB | 72K | 80.0% | 72.8% | 256K | 2.80 / 1.16 | 172.9 / 128.8 |
| unsloth/Qwen3-VL-8B-Thinking-bnb-4bit | 9.36 GiB | 9.23 GiB | 93K | 79.0% | 73.4% | 256K | 9.58 / 3.31 | 43.4 / 42.3 |
| Qwen/Qwen2.5-VL-3B-Instruct | 9.36 GiB | 9.06 GiB | 125K | 79.0% | 71.9% | 125K | 0.15 / 0.24 | 13.5 / 20.6 |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q8_K_XL | 28.65 GiB | 11.06 GiB | 74K | 76.0% | 73.8% | 256K | 3.25 / 0.95 | 196.6 / 165.5 |
| Qwen/Qwen3-VL-2B-Instruct | 6.06 GiB | 6.42 GiB | 163K | 72.0% | 77.3% | 256K | 0.15 / 0.16 | 31.8 / 30.3 |
| Qwen/Qwen3-VL-4B-Thinking | 10.46 GiB | 10.83 GiB | 73K | 77.0% | 72.2% | 256K | 10.70 / 2.81 | 60.1 / 58.9 |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q8_K_XL | 21.80 GiB | 8.32 GiB | 140K | 71.0% | 78.1% | 256K | 0.09 / 0.16 | 58.6 / 30.9 |
| unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit | 13.08 GiB | 10.56 GiB | 76K | 76.0% | 71.9% | 256K | 12.55 / 5.31 | 42.6 / 31.8 |
| Qwen/Qwen2-VL-2B-Instruct | 6.99 GiB | 6.12 GiB | 32K | 67.0% | 80.0% | 32K | 0.12 / 0.18 | 21.4 / 23.8 |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q4_K_XL | 26.27 GiB | 8.68 GiB | 105K | 70.0% | 75.3% | 256K | 2.33 / 0.68 | 277.5 / 217.9 |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q4_K_XL | 20.64 GiB | 7.16 GiB | 160K | 67.0% | 76.1% | 256K | 0.08 / 0.15 | 58.8 / 31.5 |
| unsloth/Qwen3-VL-4B-Thinking-1M-GGUF:UD-Q4_K_XL | 26.27 GiB | 8.68 GiB | 105K | 70.0% | 72.6% | 256K | 2.72 / 0.84 | 279.5 / 232.9 |
| Qwen/Qwen3-VL-2B-Thinking | 6.28 GiB | 6.45 GiB | 163K | 70.0% | 68.0% | 256K | 13.22 / 2.93 | 73.3 / 69.0 |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q4_K_XL | 23.71 GiB | 10.23 GiB | 107K | 64.0% | 69.5% | 256K | 3.52 / 1.70 | 239.8 / 123.3 |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q8_K_XL | 25.43 GiB | 11.95 GiB | 77K | 62.0% | 66.8% | 256K | 5.35 / 0.93 | 169.9 / 273.4 |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q8_K_XL | 21.82 GiB | 8.35 GiB | 139K | 53.0% | 68.6% | 256K | 3.20 / 0.81 | 346.5 / 292.7 |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q4_K_XL | 20.65 GiB | 7.18 GiB | 159K | 50.0% | 67.1% | 256K | 2.47 / 0.72 | 467.4 / 385.4 |
| zmli/G-Substrate-Qwen3-VL-2B | 10.29 GiB | 10.42 GiB | 98K | 58.0% | 41.0% | 256K | 0.20 / 0.70 | 25.9 / 57.5 |

_Eligibility uses the estimated 8K model residency and Q8 KV cache plus a 1 GiB desktop reserve. Maximum context at each limit is estimated by interpolating that memory profile and is capped at the model context limit. Benchmark scores and throughput were measured at the context listed; peak GPU use can therefore exceed the edge limit. Edge-device speed at a different context may vary._

## VRAM and context configuration

| Model | Min / default / max VRAM (context) | Peak resident GPU memory | MTP | Runner |
|---|---|---:|---|---|
| Qwen/Qwen2.5-VL-7B-Instruct | 16.72 GiB (8K) / 20.32 GiB (125K max) / 20.32 GiB (125K max) | — GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| Qwen/Qwen3-VL-2B-Instruct | 5.49 GiB (8K) / 20.74 GiB (256K) / 20.74 GiB (256K max) | 6.06 GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| Qwen/Qwen3-VL-2B-Thinking | 5.52 GiB (8K) / 20.77 GiB (256K) / 20.77 GiB (256K max) | 6.28 GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| Qwen/Qwen3-VL-8B-Instruct | 18.04 GiB (8K) / 29.98 GiB (159K max) / 29.98 GiB (159K max) | 18.65 GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| Qwen/Qwen3-VL-8B-Thinking | 18.04 GiB (8K) / 29.98 GiB (159K max) / 29.98 GiB (159K max) | 18.86 GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| zmli/G-Substrate-Qwen3-VL-2B | 9.49 GiB (8K) / 24.74 GiB (256K) / 24.74 GiB (256K max) | 10.29 GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q4_K_XL | 9.20 GiB (8K) / 23.61 GiB (256K) / 23.61 GiB (256K max) | 23.60 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q8_K_XL | 10.27 GiB (8K) / 24.68 GiB (256K) / 24.68 GiB (256K max) | 24.68 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q8_K_XL) |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q4_K_XL | 9.30 GiB (8K) / 23.71 GiB (256K) / 23.71 GiB (256K max) | 23.71 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-2B-Thinking-GGUF:UD-Q8_K_XL | 11.02 GiB (8K) / 25.43 GiB (256K) / 25.43 GiB (256K max) | 25.43 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q8_K_XL) |
| unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL | 23.89 GiB (8K) / 29.96 GiB (130K max) / 29.96 GiB (130K max) | 29.86 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL | 14.11 GiB (8K) / 29.95 GiB (220K max) / 29.95 GiB (220K max) | 23.07 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q8_K_XL) |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q4_K_XL | 10.22 GiB (8K) / 28.75 GiB (256K) / 28.75 GiB (256K max) | 28.74 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q8_K_XL | 14.08 GiB (8K) / 30.00 GiB (221K max) / 30.00 GiB (221K max) | 23.04 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q8_K_XL) |
| Qwen/Qwen2-VL-2B-Instruct | 5.19 GiB (8K) / 5.56 GiB (32K max) / 5.56 GiB (32K max) | 6.99 GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| Qwen/Qwen2-VL-7B-Instruct | 16.67 GiB (8K) / 17.41 GiB (32K max) / 17.41 GiB (32K max) | 18.21 GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| Qwen/Qwen2.5-VL-3B-Instruct | 8.13 GiB (8K) / 10.45 GiB (125K max) / 10.45 GiB (125K max) | 9.36 GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| Qwen/Qwen3-VL-4B-Instruct | 9.90 GiB (8K) / 29.51 GiB (256K) / 29.51 GiB (256K max) | 10.46 GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| Qwen/Qwen3-VL-4B-Thinking | 9.90 GiB (8K) / 29.51 GiB (256K) / 29.51 GiB (256K max) | 10.46 GiB | no (checkpoint has no MTP weights) | Transformers (model card) |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q4_K_XL | 6.23 GiB (8K) / 20.64 GiB (256K) / 30.00 GiB (417K max) | 20.64 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:UD-Q8_K_XL | 7.39 GiB (8K) / 21.80 GiB (256K) / 30.00 GiB (397K max) | 21.80 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q8_K_XL) |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q4_K_XL | 6.25 GiB (8K) / 20.66 GiB (256K) / 29.96 GiB (416K max) | 20.65 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-2B-Thinking-1M-GGUF:UD-Q8_K_XL | 7.42 GiB (8K) / 21.83 GiB (256K) / 29.96 GiB (396K max) | 21.82 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q8_K_XL) |
| unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF:UD-Q4_K_XL | 20.93 GiB (8K) / 30.00 GiB (190K max) / 30.00 GiB (190K max) | 26.90 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL | 20.93 GiB (8K) / 30.00 GiB (190K max) / 30.00 GiB (190K max) | 26.90 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL | 20.95 GiB (8K) / 29.97 GiB (189K max) / 29.97 GiB (189K max) | 26.93 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-32B-Instruct-GGUF:UD-Q4_K_XL | 22.56 GiB (8K) / 30.00 GiB (64K max) / 30.00 GiB (64K max) | 31.16 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-32B-Thinking-GGUF:UD-Q4_K_XL | 22.56 GiB (8K) / 30.00 GiB (64K max) / 30.00 GiB (64K max) | 31.15 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-4B-Instruct-1M-GGUF:UD-Q4_K_XL | 7.74 GiB (8K) / 26.27 GiB (256K) / 29.93 GiB (305K max) | 26.26 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q4_K_XL | 7.74 GiB (8K) / 26.27 GiB (256K) / 26.27 GiB (256K max) | 26.26 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL | 10.13 GiB (8K) / 28.66 GiB (256K) / 28.66 GiB (256K max) | 28.65 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q8_K_XL) |
| unsloth/Qwen3-VL-4B-Thinking-1M-GGUF:UD-Q4_K_XL | 7.75 GiB (8K) / 26.28 GiB (256K) / 29.94 GiB (305K max) | 26.27 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q4_K_XL | 7.75 GiB (8K) / 26.28 GiB (256K) / 26.28 GiB (256K max) | 26.27 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-4B-Thinking-GGUF:UD-Q8_K_XL | 10.13 GiB (8K) / 28.65 GiB (256K) / 28.65 GiB (256K max) | 28.65 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q8_K_XL) |
| unsloth/Qwen3-VL-8B-Instruct-1M-GGUF:UD-Q4_K_XL | 10.25 GiB (8K) / 28.77 GiB (256K) / 29.97 GiB (272K max) | 28.77 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL | 10.26 GiB (8K) / 28.79 GiB (256K) / 28.79 GiB (256K max) | 28.78 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen3-VL-8B-Thinking-1M-GGUF:UD-Q4_K_XL | 10.26 GiB (8K) / 28.79 GiB (256K) / 29.98 GiB (272K max) | 28.78 GiB | no (GGUF has no MTP layers) | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q4_K_XL | 22.56 GiB (8K) / 30.00 GiB (64K max) / 30.00 GiB (64K max) | 31.11 GiB | not applicable | llama.cpp (UD-Q4_K_XL) |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q5_K_XL | 25.97 GiB (8K) / 29.96 GiB (38K max) / 29.96 GiB (38K max) | 29.53 GiB | not applicable | llama.cpp (UD-Q5_K_XL) |
| unsloth/Qwen2.5-VL-72B-Instruct-GGUF:UD-IQ2_XXS | 28.49 GiB (8K) / 29.98 GiB (17K max) / 29.98 GiB (17K max) | 30.19 GiB | not applicable | llama.cpp (UD-IQ2_XXS) |
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | 28.10 GiB (8K) / 29.93 GiB (19K max) / 29.93 GiB (19K max) | 29.66 GiB | not applicable | llama.cpp (IQ2_XXS) |
| unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit | 9.63 GiB (8K) / 29.24 GiB (256K) / 29.24 GiB (256K max) | 13.08 GiB | no (checkpoint has no MTP weights) | Transformers (Unsloth BNB 4-bit model card) |
| unsloth/Qwen3-VL-8B-Thinking-bnb-4bit | 8.30 GiB (8K) / 27.92 GiB (256K) / 27.92 GiB (256K max) | 9.36 GiB | no (checkpoint has no MTP weights) | Transformers (Unsloth BNB 4-bit model card) |

## Notes

- GGUF checkpoints use their publisher's llama.cpp path and Q8_0 K/V cache; MTP is requested for Qwen3-VL models and the server log records whether a model head was found.
- Original Qwen checkpoints follow their model-card Transformers path. G-Substrate uses the Transformers image-text class for its Qwen3-VL config because its card's short AutoModelForCausalLM snippet demonstrates text loading only. These use the HQQ 8-bit KV cache where supported.
- Each task uses the same deterministic sample IDs across models. Results are small-sample selection evidence, not leaderboard scores.
- Dataset images are passed at their stored resolution; the selected model runner applies its own published image preprocessing.
- VRAM is peak whole-GPU usage sampled during the task, including the desktop baseline.
- Output tokens per second is output token count divided by end-to-end generation-call time, including image/prompt prefill.
- Context-tier VRAM values are estimates from model residency and the checkpoint KV dimensions using Q8 cache; maximum context is capped by the 30 GiB budget and the checkpoint limit.
- llama.cpp uses the largest requested context that also completes a real-image warmup under its 30 GiB fit target; it is not a claim that the dataset filled that context.
- Context tiers show 8K / 256K / 1M targets; a cell marked `max` reports the estimated last context that fits below that target.
- Parse failures are counted in the task table and score as incorrect answers.
- G-Substrate is a full SFT fine-tune of Qwen3-VL-2B-Instruct trained on scene/event/molecular graphs and graph reasoning, with a unified graph schema across modalities.
