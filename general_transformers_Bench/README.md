# General Transformers VLM benchmark

This group reuses the prepared examples, prompts, and scorers from
`qwenVL_Bench`. It keeps model-specific wrappers in `model_adapters.py`.

## Run

Run `./run_benchmark.sh` in host context. The launcher runs `release-local`
before each model, then runs PaliGemma and Florence serially. Each model must
pass one real-image AI2D answer-format check and one real-image TextVQA check
before any full-suite tasks start. Passing models run all eight shared tasks
with the plan's sample counts. Preflight examples are reused in the scored
results. A failed preflight leaves the model unscored rather than reporting a
partial suite. Each invocation writes under a timestamped `results/runs/`
directory. The aggregate report and site update only when both current model
runs pass their preflights and produce result files.

## PaliGemma 2 mix-224

Use the cached `google/paligemma2-3b-mix-224` checkpoint with
`PaliGemmaForConditionalGeneration` and `PaliGemmaProcessor`. Prefix each unchanged
shared question with `answer en `. Record the fixed 224-pixel input resolution.

## Florence-2

Florence uses the installed Python 3.11 environment under
`environments/florence/.venv`. It pins Transformers 4.44.2 and imports the
checkpoint-bundled implementation from the exact local snapshot. This avoids
Transformers' static import check misclassifying the model's guarded optional
`flash_attn` imports; eager attention remains available. The runner promotes its experimental `<VQA>`
prompt to shared scoring after both real-image preflights pass. The exact
installed package versions are in
`environments/florence/environment.json`; the resolved package list is in
`environments/florence/requirements.lock.txt`. Do not score Florence or merge
it into a completed comparison before that check.

The implementation plan proposed `<VQA>` for shared image questions. The
cached processor does not register that token or task. The parser therefore
falls back to plain text, while raw and postprocessed outputs remain available
for diagnosis. The runner checks real-image answer formatting and nonempty
VQA text before promoting this experimental route. No Florence-only tasks
enter the shared score.

## Environment setup

Use `environments/florence/python.sh` for offline Python commands in the
Florence environment. PaliGemma stays on the main Python 3.13 environment.
Do not load a model until the repository's `release-local` gate reports
success.
