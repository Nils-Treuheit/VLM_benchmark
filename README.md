# Local VLM benchmark

This repository compares vision-language models already available on the local system. The suites share fixed image tasks where their capabilities overlap, and report results separately by model family.

## Benchmark reports

| Suite | Status | Report |
|---|---|---|
| Qwen-VL | Complete | [Open report](qwenVL_Bench/) |
| LLaVA | Complete | [Open report](llava_Bench/) |
| Cosmos | Complete | [Open report](cosmos_Bench/) |
| General Transformers | Complete for the current supported candidate set | [Open report](general_transformers_Bench/) |
| Local foundation and GGUF | Complete for the current cached and runnable candidates | [Open report](local_foundation_Bench/) |
| Phi vision | Complete | [Open report](phi_Bench/) |
| Video VLMs | Complete | [Open report](video_Bench/) |
| Grounding specialists | Core suite and OPAL complete; LocateAnything, GroundAnything and YOLOE-26-L-seg cover all 100 RF100-VL subsets | [Open report](grounding_Bench/) |

The report pages use fixed local samples and support model selection on this system. They are not leaderboard results. Each report describes its tasks, sample sizes, scoring rules, model coverage, and runtime limits.

## Running benchmarks

Runs require prepared datasets and model checkpoints in the local Hugging Face cache. Start from the README in the relevant benchmark folder. Model inference runs serially on the host GPU; benchmark commands do not download model weights.

### Queue jobs through the local API

Start the queue service with `./.venv/bin/python benchmark_service.py serve`. It binds to `127.0.0.1:8765` and runs one queued command at a time. The worker checks GPU 0 every five seconds and starts the first job when VRAM use is below 1.5 GiB.

### Submit a command

Send a ready-to-run command string. The service splits it into arguments without starting a shell, captures stdout and stderr, and stores them with the job. Use `{result_dir}` where the command needs a unique output directory. Set `venv_path` to run Python commands with that environment's interpreter.

This request runs the Qwen image scene-graph benchmark across all datasets:

```sh
curl -X POST http://127.0.0.1:8765/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "benchmark":"qwen-image-sgg",
    "args":["--dataset","all","--wait-for-gpu"],
    "venv_path":"/mnt/HDD1/Project_Code/VLM_benchmark/.venv"
  }'
```

The command runs from the project root and inherits the service environment, including `HF_HOME`. Keep the model and dataset paths in the command or the benchmark's environment.

### Request fields

| Field | Required | Meaning |
|---|---|---|
| `command` | One of `command` or `benchmark` | Ready-to-run command. Use the virtual environment's Python path directly. |
| `benchmark` | One of `command` or `benchmark` | Optional registered runner ID for the legacy `args` form. |
| `args` | No | CLI arguments for a registered runner. |
| `venv_path` | No | Virtual environment for registered runners or Python commands. The service uses its `bin/python`. |

The API returns HTTP 202 and a job ID. The worker checks GPU 0 as soon as it receives a job. When used VRAM is below 1.5 GiB, it starts the command and marks the job `running`. Poll `GET /v1/jobs/<id>` for status, queue position when queued, process PID, exit code, logs, and result paths; queued responses include `queue_position`. Read `GET /v1/jobs/<id>/logs` and `/stderr` for output and errors. Filter the pending queue with `GET /v1/jobs?status=queued` and active work with `GET /v1/jobs?status=running`. Job records and logs remain under `.benchmark_service/`.

The command may include ordinary arguments and quoted values. Shell operators such as pipes and redirects are not interpreted; the service captures output itself.

The image suites reuse task definitions and scoring implemented in [Qwen-VL](qwenVL_Bench/README.md). The [implementation plan index](implementation_plans/README.md) tracks active work; completed suite reports retain their model inventories and results.

The grounding benchmark's scope, model matrix, remaining queue work, and data prerequisites are tracked in the repository's [grounding capability plan](PLAN.md).

## Report hosting

Each published report is the `index.html` at its benchmark folder root. The repository [landing page](index.html) links to available reports. A GitHub Pages deployment should preserve folder names so each suite is available under `/<benchmark-folder>/`.
