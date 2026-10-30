# GRAB-Lite graph analysis benchmark

The full official 500-example split was scored with normalized exact-match accuracy.

| Model | Overall | Properties | Functions | Series | Transforms | Real | Completed | Peak GPU | Mean seconds | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-VL-7B-Instruct | 12.6% | 16.0% | 18.0% | 8.0% | 7.0% | 14.0% | 500/500 | 23.89 GiB | 1.64 | 2.2 |
| Qwen/Qwen3-VL-2B-Instruct | 6.2% | 12.0% | 4.0% | 3.0% | 4.0% | 8.0% | 500/500 | 9.60 GiB | 1.03 | 13.9 |
| zmli/G-Substrate-Qwen3-VL-2B | 4.4% | 6.0% | 3.0% | 5.0% | 2.0% | 6.0% | 500/500 | 15.30 GiB | 3.70 | 1.1 |
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL | 14.0% | 21.0% | 10.0% | 12.0% | 6.0% | 21.0% | 500/500 | 28.53 GiB | 2.13 | 1.7 |