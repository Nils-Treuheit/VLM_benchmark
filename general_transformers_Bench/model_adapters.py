"""Model-specific loaders and prompt/output adapters for the shared suite.

These adapters preserve the shared benchmark wording. They add only the
architecture-required task prefix or model input wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PALIGEMMA_ID = "google/paligemma2-3b-mix-224"
FLORENCE_ID = "microsoft/Florence-2-large-ft"
FLORENCE_VQA_TOKEN = "<VQA>"


@dataclass(frozen=True)
class PromptSpec:
    text: str
    task_token: str | None = None
    prefix: str = "none"
    scoring_eligible: bool = True
    route_note: str | None = None


def prompt_for(model_id: str, task: str, shared_prompt: str) -> PromptSpec:
    """Return the unchanged shared prompt with only the model-required wrapper."""
    if model_id == PALIGEMMA_ID:
        return PromptSpec(f"answer en {shared_prompt}", prefix="answer en")
    if model_id == FLORENCE_ID:
        # The implementation plan proposed this VQA wrapper, but the cached
        # processor does not register <VQA>. Keep any output experimental.
        return PromptSpec(
            f"{FLORENCE_VQA_TOKEN} {shared_prompt}",
            task_token=FLORENCE_VQA_TOKEN,
            prefix=FLORENCE_VQA_TOKEN,
            scoring_eligible=False,
            route_note="<VQA> is not registered by the cached Florence processor",
        )
    return PromptSpec(shared_prompt, prefix="none")


def load_paligemma(snapshot: str | Path, *, max_memory: dict[Any, int] | None = None):
    """Load the exact cached PaliGemma 2 mix-224 checkpoint under Transformers."""
    import torch
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    model = PaliGemmaForConditionalGeneration.from_pretrained(str(snapshot), **kwargs)
    processor = PaliGemmaProcessor.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    return model, processor


def load_florence(snapshot: str | Path, *, max_memory: dict[Any, int] | None = None):
    """Load Florence through its checkpoint-bundled implementation.

    This route is isolated from the main Transformers 5.x environment. A
    successful load still requires real GPU generation before Florence is
    promoted into scored benchmark results.
    """
    import importlib
    import sys
    import types

    import torch

    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "eager",
    }
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    snapshot = Path(snapshot).resolve()
    required = ("configuration_florence2.py", "modeling_florence2.py", "processing_florence2.py")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Florence snapshot is missing custom code: {', '.join(missing)}")

    # Transformers 4.44's static remote-code import checker flags the model's
    # guarded optional flash_attn imports as mandatory. Import the exact cached
    # snapshot modules so their own eager-attention fallback remains available.
    package_name = f"_cached_florence2_{snapshot.name}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(snapshot)]
        sys.modules[package_name] = package
    model_code = importlib.import_module(f"{package_name}.modeling_florence2")
    processor_code = importlib.import_module(f"{package_name}.processing_florence2")
    model = model_code.Florence2ForConditionalGeneration.from_pretrained(
        str(snapshot), **kwargs
    )
    processor = processor_code.Florence2Processor.from_pretrained(
        str(snapshot), local_files_only=True
    )
    model.eval()
    return model, processor


def generate_paligemma(model, processor, image, spec: PromptSpec, *, max_new_tokens: int = 256):
    """Generate and decode only the completion after PaliGemma's input prompt."""
    inputs = processor(images=image.convert("RGB"), text=spec.text, return_tensors="pt")
    inputs = inputs.to(model.device)
    input_tokens = int(inputs.input_ids.shape[-1])
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    completion_ids = output_ids[:, input_tokens:]
    text = processor.batch_decode(
        completion_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text, input_tokens, int(completion_ids.shape[-1]), None


def generate_florence(model, processor, image, spec: PromptSpec, *, max_new_tokens: int = 256):
    """Generate experimental VQA text and retain raw and postprocessed output."""
    if spec.task_token != FLORENCE_VQA_TOKEN:
        raise ValueError(f"unsupported Florence task token: {spec.task_token!r}")
    inputs = processor(text=spec.text, images=image.convert("RGB"), return_tensors="pt")
    inputs = inputs.to(model.device, torch.float16)
    input_tokens = int(inputs.input_ids.shape[-1])
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=3,
    )
    # Florence is encoder-decoder, so decode the complete generated sequence.
    raw_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        raw_text,
        task=spec.task_token,
        image_size=(image.width, image.height),
    )
    answer = parsed.get(spec.task_token, parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(answer, str):
        raise TypeError(f"Florence VQA parser returned {type(answer).__name__}, expected text")
    registered_tasks = getattr(processor, "tasks_answer_post_processing_type", {})
    return answer.strip(), input_tokens, int(output_ids.shape[-1]), {
        "raw_generated_text": raw_text,
        "parsed_result": parsed,
        "parser_mode": "task-registered" if spec.task_token in registered_tasks else "fallback-pure-text",
    }


def generate(model_id: str, model, processor, image, task: str, shared_prompt: str):
    """Dispatch the model-specific prompt and generation path."""
    spec = prompt_for(model_id, task, shared_prompt)
    if model_id == PALIGEMMA_ID:
        answer, input_tokens, output_tokens, postprocess_info = generate_paligemma(
            model, processor, image, spec
        )
    elif model_id == FLORENCE_ID:
        answer, input_tokens, output_tokens, postprocess_info = generate_florence(
            model, processor, image, spec
        )
    else:
        raise ValueError(f"no specialized adapter for {model_id}")
    return {
        "prompt": spec.text,
        "prompt_prefix": spec.prefix,
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "postprocess": postprocess_info,
        "scoring_eligible": spec.scoring_eligible,
        "route_note": spec.route_note,
    }
