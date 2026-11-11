# OPAL Fine-Grained Classification Benchmark

OPAL uses ImageNet-pretrained ConvNeXt-Tiny, a normalized prototype distance classifier and 30 epochs on the official train/test splits. Each comparison panel contains 200 class-covering test images, sampled with seed 3407. Timing includes image open, RGB conversion, preprocessing, device transfer, forward pass and argmax. Model load and warm-up are excluded from query latency and reported separately.

| Dataset | Classes | Full test | Full test top-1 | Panel top-1 | Mean ms | Median ms | p95 ms | Max ms | Load s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUB-200-2011 | 200 | 5,794 | 81.96% | 80.50% | 4.28 | 4.17 | 5.43 | 7.55 | 0.30 |
| Stanford Cars | 196 | 8,041 | 89.60% | 91.00% | 5.24 | 4.25 | 10.87 | 40.83 | 0.28 |
| Oxford-IIIT Pets | 37 | 3,669 | 93.02% | 92.50% | 8.67 | 8.13 | 17.61 | 23.11 | 0.28 |
| Stanford Dogs | 120 | 8,580 | 85.94% | 87.50% | 3.57 | 3.42 | 4.58 | 8.55 | 0.27 |
| Oxford Flowers 102 | 102 | 6,149 | 95.38% | 96.50% | 4.48 | 4.38 | 5.46 | 9.05 | 0.29 |

## Grounding VLM zero-shot classification

The grounding VLMs use the same fixed 200-image panel per dataset. Their query latency includes image read and encoding, the generation request, and output parsing; it excludes initial model or service load and the one-query warm-up. OPAL timing includes image open, RGB conversion, preprocessing, device transfer, forward pass and argmax. The boundaries are similar but not identical, so speed ratios are indicative. VLM answers must identify one exact class label. The strict parser rejects lists of candidate labels instead of treating the first listed class as a prediction.

| Model | Dataset | Panel accuracy | Valid single-label responses | Mean query ms | Median ms | p95 ms | Max ms | Load s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LocateAnything-3B | CUB-200-2011 | 0.00% | 0/200 | 247.19 | 240.00 | 308.48 | 364.57 | 46.13 |
| LocateAnything-3B | Stanford Cars | 0.00% | 1/200 | 446.92 | 384.36 | 709.75 | 3,409.11 | 46.13 |
| LocateAnything-3B | Oxford-IIIT Pets | 0.00% | 0/200 | 213.29 | 208.24 | 235.89 | 800.98 | 46.13 |
| LocateAnything-3B | Stanford Dogs | 0.00% | 0/200 | 219.79 | 216.35 | 245.49 | 326.14 | 46.13 |
| LocateAnything-3B | Oxford Flowers 102 | 0.00% | 0/200 | 249.89 | 250.03 | 287.63 | 323.16 | 46.13 |
| GroundAnything-VLM | CUB-200-2011 | 3.00% | 189/200 | 337.26 | 302.40 | 431.03 | 979.27 | 128.11 |
| GroundAnything-VLM | Stanford Cars | 3.50% | 147/200 | 512.51 | 465.92 | 717.81 | 1,275.53 | 128.11 |
| GroundAnything-VLM | Oxford-IIIT Pets | 3.00% | 106/200 | 545.96 | 243.81 | 930.91 | 1,145.52 | 128.11 |
| GroundAnything-VLM | Stanford Dogs | 3.50% | 138/200 | 362.45 | 249.12 | 946.01 | 1,011.18 | 128.11 |
| GroundAnything-VLM | Oxford Flowers 102 | 2.50% | 169/200 | 400.17 | 268.67 | 978.43 | 1,036.47 | 128.11 |

LocateAnything completed all 1,000 requests, but returned multiple candidate class names rather than one answer for 999 images. Its raw outputs and strict reparsed records are preserved under `results/opal_vlm/locateanything/`. The single valid answer was incorrect. GroundAnything returned a valid single label for 749/1,000 images; accuracy over all panel images ranged from 2.5% to 3.5%. Across matched panels, LocateAnything was 24.6–85.3× slower and GroundAnything 63.0–101.5× slower than the trained OPAL classifier. Both grounding VLMs perform poorly on this classification transfer. The initial GroundAnything service load took 128.11 seconds; load time is excluded from query latency.

## Protocol and status

All five training and full-test evaluation runs are complete. SPC metrics are available for all datasets; the Pets SPC recovery covers all 3,669 test images and records mean SPC 0.00804. All five OPAL fixed-panel query timing runs are complete for LocateAnything and GroundAnything. LocateAnything completed the shared panels with zero accurate valid class answers. GroundAnything achieved 2.5%–3.5% top-1 accuracy across the panels with 0.337–0.546 seconds mean query latency, excluding its initial service load.

## Artifacts

The repository publishes these aggregate results only. Checkpoints, per-image predictions, and raw VLM records remain local and are not uploaded.

- Checkpoints and full-split metrics: `classification_Bench/results/opal_training/`
- Per-image panel predictions and timing: `classification_Bench/results/opal_timing/`
- Grounding VLM classification records: `classification_Bench/results/opal_vlm/`
- Panel manifest: `classification_Bench/data/opal_test_panels.json`
- Training runner: `classification_Bench/run_opal_training.py`
- OPAL timing runner: `classification_Bench/opal_inference_latency.py`
- VLM panel runner: `classification_Bench/opal_vlm_benchmark.py`
