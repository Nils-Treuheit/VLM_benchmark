#!/usr/bin/env python3
"""Grounding-specialist inference adapters.

LocateAnything runs from its pinned local Transformers snapshot. GroundAnything
uses its documented model-specific OpenAI-compatible SGLang endpoint.
"""

from __future__ import annotations

import base64
import inspect
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import scoring

LOCATE_ID = "nvidia/LocateAnything-3B"
LOCATE_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
GROUND_ID = "GroundingPI/GroundAnything-VLM"
GROUND_REVISION = "0f3cf22027493b4f97f342dae1fb298c38b68d27"
DEFAULT_HF_HOME = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface")
GROUND_BASE_URL = "http://127.0.0.1:8102/v1"


def snapshot_for(model_id: str, hf_home: Path | None = None) -> tuple[Path, str]:
    hf_home = hf_home or Path(__import__("os").environ.get("HF_HOME", DEFAULT_HF_HOME))
    expected = {
        LOCATE_ID: LOCATE_REVISION,
        GROUND_ID: GROUND_REVISION,
    }
    if model_id not in expected:
        raise ValueError(f"unsupported model: {model_id}")
    snapshot = hf_home / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / expected[model_id]
    index_path = snapshot / "model.safetensors.index.json"
    if not snapshot.is_dir() or not index_path.is_file():
        raise FileNotFoundError(f"pinned local model snapshot is incomplete: {snapshot}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    missing = [name for name in shards if not (snapshot / name).is_file()]
    if not shards or missing:
        raise FileNotFoundError(f"incomplete weight shards in {snapshot}: {missing or 'no indexed shards'}")
    return snapshot, expected[model_id]


def prompt_for(model_id: str, task: str, row: dict[str, Any]) -> str:
    if task in {"vqa_count", "vqa_truth", "vqa_description", "textvqa"}:
        instruction = ("Read the text in the image as needed and answer the question. "
                       "Return only the shortest exact answer, with no explanation or alternatives."
                       if task == "textvqa" else
                       "Answer the question based on the image. Return only the shortest direct answer, "
                       "with no explanation or alternatives.")
        return f"{row['question']}\n{instruction}"
    if model_id == LOCATE_ID:
        if task == "refcocog":
            return f"Locate a single instance that matches the following description: {row['phrase']}."
        if task == "coco":
            return "Locate all the instances that matches the following description: " + "</c>".join(c["name"] for c in row["categories"]) + "."
        if task == "screenspot_pro":
            return f"Point to: {row['instruction']}."
        if task == "hiertext":
            return "Detect all the text in box format."
    elif model_id == GROUND_ID:
        if task == "refcocog":
            return f"Locate the target referred to by the following description: {row['phrase']}."
        if task == "coco":
            return "Locate all the instances that match the following categories: " + "</c>".join(c["name"] for c in row["categories"]) + "."
        if task == "screenspot_pro":
            return f"Point to the UI element to click for the following instruction: {row['instruction']}."
        if task == "hiertext":
            return "OCR task detect all the text in box format."
    raise ValueError(f"unsupported task/model pair: {task}/{model_id}")


class LocateAnythingAdapter:
    def __init__(self, snapshot: Path):
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.modeling_utils import PreTrainedModel

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True)
        # LocateAnything's pinned remote code predates Transformers 5's
        # allow_all_kernels argument. Adapt its cached override in memory while
        # forwarding the new argument to the upstream implementation.
        config = AutoConfig.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True)
        serialized_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        saved_rope_theta = serialized_config.get("text_config", {}).get("rope_theta")
        if not hasattr(config.text_config, "rope_theta"):
            if saved_rope_theta is None:
                raise RuntimeError("pinned LocateAnything config has no legacy Qwen2 rope_theta value")
            # Transformers 5 normalizes this legacy field into rope_parameters;
            # the pinned custom Qwen2 implementation still reads rope_theta.
            config.text_config.rope_theta = saved_rope_theta
        model_reference = config.auto_map.get("AutoModel")
        if isinstance(model_reference, (list, tuple)):
            model_reference = model_reference[-1]
        def patch_attention_override(model_class: type, base_name: str) -> None:
            custom_base = next(
                (cls for cls in model_class.__mro__ if cls.__name__ == base_name), None,
            )
            if custom_base is None:
                raise RuntimeError(f"{base_name} was not found in the pinned model snapshot")
            original_check = custom_base._check_and_adjust_attn_implementation
            if "allow_all_kernels" not in inspect.signature(original_check).parameters:
                def compatible_check(self, attn_implementation, is_init_check=False, allow_all_kernels=False):
                    if attn_implementation == "magi":
                        return "magi"
                    upstream_check = super(custom_base, self)._check_and_adjust_attn_implementation
                    upstream_parameters = inspect.signature(upstream_check).parameters
                    if "allow_all_kernels" in upstream_parameters:
                        return upstream_check(
                            attn_implementation, is_init_check, allow_all_kernels=allow_all_kernels,
                        )
                    return upstream_check(attn_implementation, is_init_check)

                custom_base._check_and_adjust_attn_implementation = compatible_check

            legacy_tied_keys = model_class.__dict__.get("_tied_weights_keys")
            if isinstance(legacy_tied_keys, list) and hasattr(PreTrainedModel, "get_expanded_tied_weights_keys"):
                if legacy_tied_keys != ["lm_head.weight"] or base_name != "Qwen2PreTrainedModel":
                    raise RuntimeError(f"unsupported legacy tied-weight declaration on {model_class.__name__}")
                model_class._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

        model_class = get_class_from_dynamic_module(
            model_reference, str(snapshot), local_files_only=True,
        )
        patch_attention_override(model_class, "LocateAnythingPreTrainedModel")
        # Patch the Qwen class imported by LocateAnything's actual dynamic module
        # so this is the same Python class used during model construction.
        model_module = sys.modules[model_class.__module__]
        qwen_model_class = getattr(model_module, "Qwen2ForCausalLM")
        patch_attention_override(qwen_model_class, "Qwen2PreTrainedModel")
        original_init = model_class.__init__
        if (not hasattr(model_class, "all_tied_weights_keys")
                and hasattr(PreTrainedModel, "get_expanded_tied_weights_keys")):
            def init_with_tied_weight_metadata(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=True)

            model_class.__init__ = init_with_tied_weight_metadata
        self.model = AutoModel.from_pretrained(
            str(snapshot), trust_remote_code=True, local_files_only=True,
            config=config, torch_dtype=torch.bfloat16,
        ).to("cuda:0").eval()
        self.device = "cuda:0"
        self.mode = "hybrid"
        self.precision = "bfloat16"
        self.backend = "Transformers"

    def generate(self, image_path: Path, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt},
        ]}]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        encoded = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        pixel_values = encoded["pixel_values"].to(self.torch.bfloat16)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                image_grid_hws=encoded.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode="hybrid",
                temperature=0,
                do_sample=False,
                verbose=False,
            )
        elapsed = time.perf_counter() - started
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        raw = output[0] if isinstance(output, tuple) else output
        raw = raw if isinstance(raw, str) else self.tokenizer.decode(raw, skip_special_tokens=False)
        output_tokens = len(self.tokenizer.encode(raw, add_special_tokens=False))
        return {"raw_output": raw, "latency_s": elapsed, "input_tokens": prompt_tokens,
                "output_tokens": output_tokens, "finish_reason": "length" if output_tokens >= max_new_tokens else "stop",
                "parse": scoring.parse_locateanything(raw)}


class GroundAnythingAdapter:
    """Serial image requests to the official model-specific SGLang server."""

    def __init__(self, base_url: str = GROUND_BASE_URL, model: str = "groundinganything-vlm", timeout_s: float = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.mode = "autoregressive; release BF16/eager SGLang profile"
        self.precision = "bfloat16"
        self.backend = "GroundAnything custom SGLang"

    def generate(self, image_path: Path, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        image_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        request_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_data}"}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0,
            "max_tokens": max_new_tokens,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"GroundAnything request failed: {type(exc).__name__}: {exc}") from exc
        elapsed = time.perf_counter() - started
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"GroundAnything returned no choices: {payload}")
        choice = choices[0]
        content = choice.get("message", {}).get("content", "")
        if isinstance(content, list):
            raw = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            raw = str(content)
        usage = payload.get("usage") or {}
        return {"raw_output": raw, "latency_s": elapsed,
                "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"), "parse": scoring.parse_groundanything(raw)}

    def model_inventory(self) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + "/models")
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read())


def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    names = ("torch", "transformers", "Pillow", "safetensors", "sglang", "openai")
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result
