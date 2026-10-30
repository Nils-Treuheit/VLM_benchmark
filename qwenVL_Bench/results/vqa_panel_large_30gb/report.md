# VQA skill panel benchmark

Generated 2026-10-29 16:26:12 +0100 with 25 fixed examples per question type.
Requests are serial, single image, and single turn. The VQA panel uses category consensus or accuracy. GRAB-Lite uses normalized exact-match accuracy over its complete 500-question split.

| Model | Question type | Score | Completed | Peak GPU | Mean seconds | Output tokens/s |
|---|---|---:|---:|---:|---:|---:|
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | vqa_truth | 95.6% | 25/25 | 29.57 GiB | 0.37 | 5.3 |
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | vqa_count | 81.6% | 25/25 | 29.57 GiB | 0.36 | 5.8 |
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | vqa_description | 59.6% | 25/25 | 29.57 GiB | 0.37 | 7.0 |
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | vqa_affordance | 4.0% | 25/25 | 29.59 GiB | 0.86 | 3.0 |
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | vqa_negation | 68.0% | 25/25 | 29.94 GiB | 0.78 | 2.6 |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q4_K_XL | vqa_truth | 81.2% | 25/25 | 30.84 GiB | 0.25 | 10.6 |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q4_K_XL | vqa_count | 15.6% | 25/25 | 30.84 GiB | 0.26 | 14.5 |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q4_K_XL | vqa_description | 43.2% | 25/25 | 30.84 GiB | 0.26 | 15.4 |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q4_K_XL | vqa_affordance | 0.0% | 25/25 | 30.87 GiB | 0.58 | 5.4 |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q4_K_XL | vqa_negation | 62.5% | 8/25 | 31.08 GiB | 0.61 | 7.0 |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q5_K_XL | vqa_truth | 81.2% | 25/25 | 29.30 GiB | 0.26 | 9.7 |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q5_K_XL | vqa_count | 31.2% | 25/25 | 29.30 GiB | 0.26 | 12.3 |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q5_K_XL | vqa_description | 51.2% | 25/25 | 29.30 GiB | 0.45 | 30.5 |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q5_K_XL | vqa_affordance | 0.0% | 25/25 | 29.32 GiB | 0.60 | 5.2 |
| unsloth/Qwen2.5-VL-32B-Instruct-GGUF:UD-Q5_K_XL | vqa_negation | 60.0% | 25/25 | 30.13 GiB | 1.05 | 25.3 |
| unsloth/Qwen2.5-VL-72B-Instruct-GGUF:UD-IQ2_XXS | vqa_truth | 89.2% | 25/25 | 29.97 GiB | 0.38 | 5.3 |
| unsloth/Qwen2.5-VL-72B-Instruct-GGUF:UD-IQ2_XXS | vqa_count | 75.2% | 25/25 | 29.97 GiB | 0.37 | 5.8 |
| unsloth/Qwen2.5-VL-72B-Instruct-GGUF:UD-IQ2_XXS | vqa_description | 66.8% | 25/25 | 29.97 GiB | 0.38 | 7.3 |
| unsloth/Qwen2.5-VL-72B-Instruct-GGUF:UD-IQ2_XXS | vqa_affordance | 4.0% | 25/25 | 30.00 GiB | 0.90 | 2.9 |
| unsloth/Qwen2.5-VL-72B-Instruct-GGUF:UD-IQ2_XXS | vqa_negation | 84.0% | 25/25 | 30.81 GiB | 0.88 | 2.3 |
| unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF:UD-Q4_K_XL | vqa_truth | 95.6% | 25/25 | 26.33 GiB | 0.12 | 16.8 |
| unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF:UD-Q4_K_XL | vqa_count | 69.6% | 25/25 | 26.33 GiB | 0.11 | 18.5 |
| unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF:UD-Q4_K_XL | vqa_description | 58.0% | 25/25 | 26.33 GiB | 0.12 | 25.8 |
| unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF:UD-Q4_K_XL | vqa_affordance | 4.0% | 25/25 | 26.33 GiB | 0.27 | 11.1 |
| unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF:UD-Q4_K_XL | vqa_negation | 56.0% | 25/25 | 26.68 GiB | 0.31 | 40.6 |
| unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL | vqa_truth | 95.6% | 25/25 | 26.32 GiB | 0.12 | 16.6 |
| unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL | vqa_count | 78.8% | 25/25 | 26.33 GiB | 0.11 | 18.3 |
| unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL | vqa_description | 62.4% | 25/25 | 26.33 GiB | 0.12 | 25.8 |
| unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL | vqa_affordance | 4.0% | 25/25 | 26.33 GiB | 0.27 | 10.3 |
| unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL | vqa_negation | 60.0% | 25/25 | 26.63 GiB | 0.30 | 40.6 |
| unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL | vqa_truth | 82.4% | 25/25 | 26.38 GiB | 0.66 | 227.9 |
| unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL | vqa_count | 75.6% | 25/25 | 26.38 GiB | 1.25 | 253.6 |
| unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL | vqa_description | 52.4% | 25/25 | 26.38 GiB | 0.66 | 237.2 |
| unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL | vqa_affordance | 12.0% | 25/25 | 26.38 GiB | 1.11 | 210.7 |
| unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL | vqa_negation | 68.0% | 25/25 | 26.62 GiB | 2.46 | 243.4 |
| unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL | vqa_truth | 77.2% | 25/25 | 26.20 GiB | 0.56 | 219.3 |
| unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL | vqa_count | 72.4% | 25/25 | 26.32 GiB | 1.38 | 250.8 |
| unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL | vqa_description | 43.6% | 25/25 | 26.32 GiB | 0.62 | 229.8 |
| unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL | vqa_affordance | 8.0% | 25/25 | 26.37 GiB | 1.04 | 202.3 |
| unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL | vqa_negation | 60.0% | 25/25 | 26.61 GiB | 2.79 | 245.9 |
| unsloth/Qwen3-VL-32B-Instruct-GGUF:UD-Q4_K_XL | vqa_truth | 83.6% | 25/25 | 30.57 GiB | 0.23 | 10.5 |
| unsloth/Qwen3-VL-32B-Instruct-GGUF:UD-Q4_K_XL | vqa_count | 84.0% | 25/25 | 30.58 GiB | 0.22 | 9.5 |
| unsloth/Qwen3-VL-32B-Instruct-GGUF:UD-Q4_K_XL | vqa_description | 63.2% | 25/25 | 30.60 GiB | 0.23 | 12.3 |
| unsloth/Qwen3-VL-32B-Instruct-GGUF:UD-Q4_K_XL | vqa_affordance | 4.0% | 25/25 | 30.72 GiB | 0.59 | 7.3 |
| unsloth/Qwen3-VL-32B-Instruct-GGUF:UD-Q4_K_XL | vqa_negation | 76.0% | 25/25 | 31.03 GiB | 0.50 | 4.0 |
| unsloth/Qwen3-VL-32B-Thinking-GGUF:UD-Q4_K_XL | vqa_truth | 61.2% | 25/25 | 30.71 GiB | 3.11 | 56.7 |
| unsloth/Qwen3-VL-32B-Thinking-GGUF:UD-Q4_K_XL | vqa_count | 70.0% | 25/25 | 30.79 GiB | 6.70 | 60.5 |
| unsloth/Qwen3-VL-32B-Thinking-GGUF:UD-Q4_K_XL | vqa_description | 48.8% | 25/25 | 30.79 GiB | 2.44 | 60.6 |
| unsloth/Qwen3-VL-32B-Thinking-GGUF:UD-Q4_K_XL | vqa_affordance | 8.0% | 25/25 | 30.79 GiB | 4.51 | 56.8 |
| unsloth/Qwen3-VL-32B-Thinking-GGUF:UD-Q4_K_XL | vqa_negation | 72.0% | 25/25 | 30.97 GiB | 10.33 | 60.3 |