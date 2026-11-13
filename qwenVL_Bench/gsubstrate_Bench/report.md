# G-Substrate cross-task subsample results

## G-Substrate Qwen3-VL 2B

Model revision: `e3c7d3c8ffd278884522739ae7bb9e05b4c18a1a`
Dataset revision: `388622518ef349efd051930bdca244fde82e7d0b`
Official repository commit: `68d00d258e2be282abca41ef4912b323f747f31c`
Decoding: `{"do_sample": true, "max_new_tokens": 4096, "sampling_seed": 42, "temperature": 0.7, "top_k": 20, "top_p": 0.8}`.

| Metric | n | Estimate | Advertised | Difference (pp) | 95% bootstrap interval | Close? |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| GAR-CT | 50 | 100.00 | 98.41 | +1.59 | [100.00, 100.00] | yes |
| GAR-CD | 50 | 96.00 | 96.97 | -0.97 | [90.00, 100.00] | yes |
| GAR-SP | 50 | 54.00 | 48.59 | +5.41 | [40.00, 68.00] | yes |
| GAR-BM | 50 | 96.00 | 94.54 | +1.46 | [90.00, 100.00] | yes |
| GAR-mean | — | 86.50 | 84.63 | +1.87 | [82.50, 90.50] | yes |
| MGD-BLEU4 | 200 | 51.03 | 51.53 | -0.50 | [46.70, 55.38] | yes |
| MGD-ROUGE-L | 200 | 69.83 | 68.47 | +1.36 | [66.28, 73.39] | yes |
| ERE-MA-S | 60 | 45.81 | 52.20 | -6.39 | [36.08, 56.54] | yes |
| ERE-MA-T | 60 | 46.74 | 42.68 | +4.06 | [38.78, 53.93] | yes |
| ERE-MA-C | 60 | 42.98 | 40.91 | +2.07 | [32.57, 53.28] | yes |
| ERE-HiE | 20 | 29.34 | 25.15 | +4.19 | [21.30, 38.59] | yes |
| SGG-PCIs-R@50 | 50 | 22.68 | 25.38 | -2.70 | — | yes |
| SGG-PCIs-mR@50 | 50 | 8.06 | 8.67 | -0.61 | — | yes |

### Inference validity

Completed rows: 600/600; errors: 0; token-cap hits: 2; empty outputs by family: `{"ERE": 0, "GAR": 0, "MGD": 0}`.
ERE parsed edge counts by task: `{"hievents": {"gold_edges": 1064, "parsed_rows": 20, "predicted_edges": 1298}, "maven_ere_causal_anchor": {"gold_edges": 127, "parsed_rows": 60, "predicted_edges": 89}, "maven_ere_subevent_anchor": {"gold_edges": 138, "parsed_rows": 60, "predicted_edges": 123}, "maven_ere_temporal_anchor": {"gold_edges": 616, "parsed_rows": 60, "predicted_edges": 644}}`.

### SGG fixed-panel evidence

The existing reviewed 50-image VG canonical-test result is included as a separate SGG panel, not merged into the G-Substrate HF test split. It scored PCIs R@50 22.68% and mR@50 8.06% against advertised 25.38% and 8.67%. It has 42/50 strict-valid rows, seven tuple-limit failures, one token-cap hit, and an independent ACCEPT WITH CAVEAT review. Its denominator is 410 ground-truth relations. No interval is reported for this fixed panel.

### Provenance and limitations

Model weights SHA-256: `8a91ac7c99a15b19363f5f726627cdfb46fdb8ac71bd8befc0468a5b5f652d4d`; chat-template SHA-256: `3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4`.
Scorer files: `{"evaluate.py": "ebd89bd1de3a89c7684421afb25deace9ab80d5d28dd2cf34dab39c1b5596e66", "event_graph_eval.py": "783fa996b508732b16c3ea8a90c189fe8a7e2974b307b1bec6e903cea6bf83c6", "graph_search_eval.py": "f04ff79352c34f2c703b20d5ec9cc6b21cc93136e6fc20c9949f58e8909dd79f", "mol_eval.py": "ce5cb8aabb9da2ef40664713d301f32580675e061cd6526668cac70e37a58330"}`.
GAR/MGD/ERE estimates use the official test split and fixed deterministic subsamples. They are not full-test reproductions. A sample is called close when its difference is at most five percentage points or the advertised value falls within its 95% bootstrap interval. No prompt or sample tuning was performed against the advertised values.

## Qwen2-VL 7B

Model revision: `eed13092ef92e448dd6875b2a00151bd3f7db0ac`
Dataset revision: `388622518ef349efd051930bdca244fde82e7d0b`
Official repository commit: `68d00d258e2be282abca41ef4912b323f747f31c`
Decoding: `{"do_sample": true, "max_new_tokens": 4096, "sampling_seed": 42, "temperature": 0.7, "top_k": 20, "top_p": 0.8}`.

| Metric | n | Estimate | Advertised | Difference (pp) | 95% bootstrap interval | Close? |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| GAR-CT | 50 | 0.00 | 98.41 | -98.41 | [0.00, 0.00] | no |
| GAR-CD | 50 | 0.00 | 96.97 | -96.97 | [0.00, 0.00] | no |
| GAR-SP | 50 | 0.00 | 48.59 | -48.59 | [0.00, 0.00] | no |
| GAR-BM | 50 | 0.00 | 94.54 | -94.54 | [0.00, 0.00] | no |
| GAR-mean | — | 0.00 | 84.63 | -84.63 | [0.00, 0.00] | no |
| MGD-BLEU4 | 200 | 2.67 | 51.53 | -48.86 | [2.04, 3.56] | no |
| MGD-ROUGE-L | 200 | 20.63 | 68.47 | -47.84 | [19.58, 21.79] | no |
| ERE-MA-S | 60 | 0.00 | 52.20 | -52.20 | [0.00, 0.00] | no |
| ERE-MA-T | 60 | 0.00 | 42.68 | -42.68 | [0.00, 0.00] | no |
| ERE-MA-C | 60 | 0.00 | 40.91 | -40.91 | [0.00, 0.00] | no |
| ERE-HiE | 20 | 4.38 | 25.15 | -20.77 | [1.53, 7.68] | no |

### Inference validity

Completed rows: 600/600; errors: 0; token-cap hits: 21; empty outputs by family: `{"ERE": 0, "GAR": 0, "MGD": 0}`.
ERE parsed edge counts by task: `{"hievents": {"gold_edges": 1064, "parsed_rows": 16, "predicted_edges": 785}, "maven_ere_causal_anchor": {"gold_edges": 127, "parsed_rows": 0, "predicted_edges": 0}, "maven_ere_subevent_anchor": {"gold_edges": 138, "parsed_rows": 0, "predicted_edges": 0}, "maven_ere_temporal_anchor": {"gold_edges": 616, "parsed_rows": 0, "predicted_edges": 0}}`.
Official Graph Facts Event Graph conversions: 19 rows; raw predictions remain unchanged in `predictions.jsonl`.

### SGG fixed-panel evidence

SGG is scored in a separate image panel for each candidate. This text-only report does not include an SGG score.

### Provenance and limitations

Model weights SHA-256: `e084a35ed082675e667bcf3a62fd861d3a5731bbcf9ac71ee63ef34facd6c4a8`; chat-template SHA-256: `a0bc6f6fc7a29a80017a433e8f03a1cc1236e838a944a2d034295a60c4f2fddb`.
Implementation source SHA-256: `{"runner": "30e227405ce21c92095fb38da2cbb9b63108eabcfacc3da87557d6ea762696f0", "scorer": "25db73c3a88a683230b79229f0c93b372aba14b63785adb5be6f074612a67400"}`.
Scorer files: `{"evaluate.py": "ebd89bd1de3a89c7684421afb25deace9ab80d5d28dd2cf34dab39c1b5596e66", "event_graph_eval.py": "783fa996b508732b16c3ea8a90c189fe8a7e2974b307b1bec6e903cea6bf83c6", "graph_search_eval.py": "f04ff79352c34f2c703b20d5ec9cc6b21cc93136e6fc20c9949f58e8909dd79f", "mol_eval.py": "ce5cb8aabb9da2ef40664713d301f32580675e061cd6526668cac70e37a58330"}`.
GAR/MGD/ERE estimates use the official test split and fixed deterministic subsamples. They are not full-test reproductions. A sample is called close when its difference is at most five percentage points or the advertised value falls within its 95% bootstrap interval. No prompt or sample tuning was performed against the advertised values.
