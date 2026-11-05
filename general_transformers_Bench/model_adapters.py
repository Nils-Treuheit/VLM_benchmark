"""Model-specific loaders and prompt/output adapters for the shared suite.

These adapters preserve the shared benchmark wording. They add only the
architecture-required task prefix or model input wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


PALIGEMMA_ID = "google/paligemma2-3b-mix-224"
FLORENCE_ID = "microsoft/Florence-2-large-ft"
MOLMO2_ID = "allenai/Molmo2-8B"
INTERNVL_ID = "OpenGVLab/InternVL3_5-30B-A3B"
LLAMA_VISION_ID = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
LLAMA_VISION_ROOT = Path(
    "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/llama3_2_vision_candidates"
)
LLAMA_VISION_SNAPSHOT = LLAMA_VISION_ROOT / "Llama-3.2-11B-Vision-Instruct-bnb-4bit"
FLORENCE_VQA_TOKEN = "<VQA>"
INTERNVL_REVISION = "ba1762d830e2aeddb3272b66ba9e0abf7e69f626"
INTERNVL35_HF_CANDIDATES = {
    "OpenGVLab/InternVL3_5-8B-Instruct": "InternVL3_5-8B-Instruct",
    "OpenGVLab/InternVL3_5-8B-MPO": "InternVL3_5-8B-MPO",
    "OpenGVLab/InternVL3_5-8B": "InternVL3_5-8B-CascadeRL",
}
INTERNVL35_CANDIDATE_IDS = tuple(INTERNVL35_HF_CANDIDATES)
INTERNVL35_CANDIDATE_ROOT = Path(
    "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/internvl3_5_candidates"
)
INTERNVL35_GGUF_CANDIDATES = {
    "bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M": {
        "directory": "InternVL3_5-30B-A3B-GGUF",
        "model": "OpenGVLab_InternVL3_5-30B-A3B-Q4_K_M.gguf",
        "mmproj": "mmproj-OpenGVLab_InternVL3_5-30B-A3B-f16.gguf",
        "repo_id": "bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF",
    },
    "bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M": {
        "directory": "InternVL3_5-14B-GGUF",
        "model": "OpenGVLab_InternVL3_5-14B-Q4_K_M.gguf",
        "mmproj": "mmproj-OpenGVLab_InternVL3_5-14B-bf16.gguf",
        "repo_id": "bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
    },
    "bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0": {
        "directory": "InternVL3_5-14B-GGUF",
        "model": "OpenGVLab_InternVL3_5-14B-Q8_0.gguf",
        "mmproj": "mmproj-OpenGVLab_InternVL3_5-14B-bf16.gguf",
        "repo_id": "bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
    },
}
INTERNVL35_GGUF_IDS = tuple(INTERNVL35_GGUF_CANDIDATES)
INTERNVL_GPU_IDS = frozenset((INTERNVL_ID, *INTERNVL35_CANDIDATE_IDS, *INTERNVL35_GGUF_IDS))


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
        "attn_implementation": "eager",
    }
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    model = PaliGemmaForConditionalGeneration.from_pretrained(str(snapshot), **kwargs)
    processor = PaliGemmaProcessor.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    return model, processor


def load_molmo2(snapshot: str | Path, *, max_memory: dict[Any, int] | None = None):
    """Load the cached Molmo2 8B checkpoint with its official Transformers path."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    model = AutoModelForImageTextToText.from_pretrained(str(snapshot), **kwargs)
    processor = AutoProcessor.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True, padding_side="left"
    )
    model.eval()
    return model, processor


def load_florence(snapshot: str | Path):
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
        "attn_implementation": "eager",
    }
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
    ).to(torch.device("cuda", torch.cuda.current_device()))
    processor = processor_code.Florence2Processor.from_pretrained(
        str(snapshot), local_files_only=True
    )
    model.eval()
    return model, processor


def load_internvl(snapshot: str | Path, *, module_cache: str | Path):
    """Load the pinned GitHub-format InternVL checkpoint with strict GPU placement."""
    import json
    import os

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    snapshot = Path(snapshot).resolve()
    cache = Path(module_cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(cache)
    # Transformers caches these module-level values at import time. Set both
    # before dynamic custom-code resolution so nested config imports stay local.
    import transformers.dynamic_module_utils as dynamic_modules
    import transformers.utils.hub as transformers_hub

    dynamic_modules.HF_MODULES_CACHE = str(cache)
    transformers_hub.HF_MODULES_CACHE = str(cache)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("_commit_hash") not in (None, INTERNVL_REVISION):
        raise RuntimeError(f"unexpected InternVL snapshot revision: {config.get('_commit_hash')}")
    if config.get("tie_word_embeddings", False) or config.get("llm_config", {}).get("tie_word_embeddings", False):
        raise RuntimeError("InternVL checkpoint enables tied embeddings unsupported by its custom loader")
    model_ref = config.get("auto_map", {}).get("AutoModel")
    if not model_ref:
        raise RuntimeError("InternVL config does not define its custom AutoModel class")
    model_class = get_class_from_dynamic_module(
        model_ref,
        str(snapshot),
        cache_dir=str(cache),
        local_files_only=True,
    )
    # This custom class predates Transformers' cached tied-weight mapping;
    # config disables tied embeddings, so an empty mapping is correct.
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model = model_class.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def load_internvl35_hf(snapshot: str | Path, *, module_cache: str | Path):
    """Load one official 8B HF-format candidate as full BF16 on cuda:0."""
    import json
    import os
    import shutil

    import torch
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    snapshot = Path(snapshot).resolve()
    cache = Path(module_cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(cache)
    import transformers.dynamic_module_utils as dynamic_modules
    import transformers.utils.hub as transformers_hub

    dynamic_modules.HF_MODULES_CACHE = str(cache)
    transformers_hub.HF_MODULES_CACHE = str(cache)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    model_ref = config.get("auto_map", {}).get("AutoModel")
    if not model_ref:
        raise RuntimeError("InternVL3.5 HF config does not define its custom AutoModel class")
    # Dynamic loading can omit transitive relative imports while creating the
    # model module. Seed its local folder with every support file first.
    module_dir = cache / "transformers_modules" / snapshot.name.replace("-", "_hyphen_")
    module_dir.mkdir(parents=True, exist_ok=True)
    for source in snapshot.glob("*.py"):
        shutil.copy2(source, module_dir / source.name)
    model_class = get_class_from_dynamic_module(
        model_ref,
        str(snapshot),
        cache_dir=str(cache),
        local_files_only=True,
    )
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model = model_class.from_pretrained(
        snapshot,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        use_flash_attn=True,
    )
    model.eval()
    return model, tokenizer


def load_llama_vision(snapshot: str | Path):
    """Load the local 4-bit Mllama checkpoint fully onto the selected CUDA device."""
    import torch
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    snapshot = Path(snapshot).resolve()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = MllamaForConditionalGeneration.from_pretrained(
        snapshot,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
        local_files_only=True,
        # Transformers 4.57 routes MllamaVisionAttention through FlashAttention
        # without the is_causal attribute expected by that integration. SDPA
        # supports the model's image path in this runtime.
        attn_implementation="sdpa",
    )
    model.eval()
    return model, processor


def generate_llama_vision(model, processor, image, spec: PromptSpec, *, max_new_tokens: int = 256):
    """Run the shared prompt through Mllama's image-aware conversation template."""
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": spec.text}],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(model.device)
    input_tokens = int(inputs.input_ids.shape[-1])
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    completion_ids = output_ids[:, input_tokens:]
    answer = processor.decode(
        completion_ids[0], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return answer, input_tokens, int(completion_ids.shape[-1]), {
        "chat_template": "Mllama image-text conversation template",
        "image_transport": "local RGB image through AutoProcessor",
    }


class LlamaCppServer:
    """One local llama.cpp server process for a single GGUF benchmark run."""

    def __init__(self, process, log_handle, *, base_url: str, context_size: int):
        from types import SimpleNamespace

        self.process = process
        self.log_handle = log_handle
        self.base_url = base_url.rstrip("/")
        self.context_size = context_size
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=context_size)
        )

    def wait_ready(self, timeout_s: float = 240.0) -> None:
        import time
        from urllib.error import URLError
        from urllib.request import urlopen

        deadline = time.monotonic() + timeout_s
        last_error = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup ({self.process.returncode}):\n"
                    + self.read_log_tail()
                )
            try:
                with urlopen(self.base_url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, URLError) as exc:
                last_error = exc
            time.sleep(0.5)
        raise TimeoutError(
            f"llama-server did not become ready in {timeout_s:.0f}s; last error={last_error}\n"
            + self.read_log_tail()
        )

    def generate(self, image, prompt: str, *, max_new_tokens: int) -> tuple[str, int, int, dict[str, Any]]:
        import base64
        import io
        import json
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG", optimize=True)
        image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "model": "internvl35-local",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
        }
        request = Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server returned HTTP {exc.code}: {details[:2000]}") from exc
        except URLError as exc:
            raise RuntimeError(f"llama-server request failed: {exc}") from exc
        try:
            content = result["choices"][0]["message"]["content"]
            usage = result["usage"]
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"llama-server returned an incomplete chat response: {result!r}") from exc
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str):
            raise RuntimeError(f"llama-server returned {type(content).__name__} message content")
        return content.strip(), input_tokens, output_tokens, {
            "backend": "llama.cpp OpenAI-compatible multimodal chat",
            "image_transport": "PNG data URL",
            "context_size": self.context_size,
            "usage_source": "llama-server response",
        }

    def read_log_tail(self, line_count: int = 60) -> str:
        self.log_handle.flush()
        try:
            text = Path(self.log_handle.name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(server log unavailable)"
        return "\n".join(text.splitlines()[-line_count:])

    def require_full_gpu_offload(self) -> tuple[int, int]:
        import re

        self.log_handle.flush()
        text = Path(self.log_handle.name).read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", text)
        if not matches:
            raise RuntimeError(
                "llama.cpp did not report a verifiable GPU layer count:\n" + self.read_log_tail()
            )
        loaded, total = map(int, matches[-1])
        if total <= 0 or loaded != total:
            raise RuntimeError(
                f"llama.cpp placed only {loaded}/{total} layers on GPU; full-GPU candidate required"
            )
        return loaded, total

    def close(self) -> None:
        import subprocess

        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log_handle.close()


def load_llama_cpp_server(
    model_path: str | Path,
    mmproj_path: str | Path,
    log_path: str | Path,
    *,
    port: int,
    context_size: int = 8192,
) -> LlamaCppServer:
    """Start llama.cpp with all transformer layers requested on the CUDA GPU."""
    import subprocess

    binary = Path(__file__).resolve().parent / "environments" / "llama_cpp" / "build" / "bin" / "llama-server"
    model_path = Path(model_path).resolve()
    mmproj_path = Path(mmproj_path).resolve()
    log_path = Path(log_path).resolve()
    for path in (binary, model_path, mmproj_path):
        if not path.is_file():
            raise FileNotFoundError(f"llama.cpp benchmark file is missing: {path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        str(binary),
        "--model", str(model_path),
        "--mmproj", str(mmproj_path),
        "--n-gpu-layers", "999",
        "--ctx-size", str(context_size),
        "--batch-size", "128",
        "--ubatch-size", "128",
        "--parallel", "1",
        "--flash-attn", "on",
        "--log-verbosity", "5",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--alias", "internvl35-local",
        "--no-webui",
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=binary.parent,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    return LlamaCppServer(
        process,
        log_handle,
        base_url=f"http://127.0.0.1:{port}",
        context_size=context_size,
    )


def _internvl_aspect_grid(
    width: int, height: int, image_size: int, max_num: int
) -> tuple[int, int]:
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (cols, rows)
            for n in range(1, max_num + 1)
            for cols in range(1, n + 1)
            for rows in range(1, n + 1)
            if 1 <= cols * rows <= max_num
        },
        key=lambda pair: pair[0] * pair[1],
    )
    best = (1, 1)
    best_diff = float("inf")
    for cols, rows in target_ratios:
        diff = abs(aspect_ratio - cols / rows)
        if diff < best_diff:
            best_diff = diff
            best = (cols, rows)
        elif diff == best_diff and width * height > 0.5 * image_size * image_size * cols * rows:
            best = (cols, rows)
    return best


def internvl_tiles(image, *, image_size: int = 448, max_num: int = 12):
    """Apply InternVL's documented dynamic tiling and optional thumbnail."""
    import torch
    from PIL import Image
    from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor
    from torchvision.transforms.functional import InterpolationMode

    image = image.convert("RGB")
    cols, rows = _internvl_aspect_grid(image.width, image.height, image_size, max_num)
    resized = image.resize(
        (cols * image_size, rows * image_size), Image.Resampling.BICUBIC
    )
    transform = Compose(
        [
            Lambda(lambda value: value.convert("RGB") if value.mode != "RGB" else value),
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    tiles = [
        transform(
            resized.crop(
                (
                    col * image_size,
                    row * image_size,
                    (col + 1) * image_size,
                    (row + 1) * image_size,
                )
            )
        )
        for row in range(rows)
        for col in range(cols)
    ]
    if len(tiles) > 1:
        tiles.append(transform(image.resize((image_size, image_size), Image.Resampling.BICUBIC)))
    return torch.stack(tiles)


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


def generate_molmo2(model, processor, image, spec: PromptSpec, *, max_new_tokens: int = 256):
    """Use Molmo2's chat template and decode only the generated completion."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": spec.text},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=prompt,
        images=[image.convert("RGB")],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    input_tokens = int(inputs.input_ids.shape[-1])
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    completion_ids = output_ids[:, input_tokens:]
    text = processor.tokenizer.decode(
        completion_ids[0], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return text, input_tokens, int(completion_ids.shape[-1]), {
        "chat_template": True,
        "completion_only_decode": True,
    }


def generate_florence(model, processor, image, spec: PromptSpec, *, max_new_tokens: int = 256):
    """Generate experimental VQA text and retain raw and postprocessed output."""
    import torch

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


def generate_internvl(model, tokenizer, image, spec: PromptSpec, *, max_new_tokens: int = 256):
    """Generate one image response through InternVL's custom chat interface."""
    import torch

    tiles = internvl_tiles(image)
    vision_parameter = next(model.vision_model.parameters())
    pixel_values = tiles.to(device=vision_parameter.device, dtype=torch.bfloat16)
    with torch.inference_mode():
        answer = model.chat(
            tokenizer,
            pixel_values,
            spec.text,
            {"max_new_tokens": max_new_tokens, "do_sample": False},
            history=None,
            return_history=False,
            num_patches_list=[len(tiles)],
        )
    answer = str(answer).strip()
    # The documented chat API returns decoded text, not token IDs. These counts
    # are tokenizer re-encodings and are explicitly labeled in provenance.
    input_tokens = len(tokenizer(spec.text, add_special_tokens=False).input_ids)
    output_tokens = len(tokenizer(answer, add_special_tokens=False).input_ids)
    return answer, input_tokens, output_tokens, {
        "image_tile_count": int(len(tiles)),
        "tile_policy": {"image_size": 448, "max_num": 12, "thumbnail": True},
        "token_count_method": "decoded_text_retokenized",
        "chat_template": "InternVL custom chat API",
    }


def generate(model_id: str, model, processor, image, task: str, shared_prompt: str):
    """Dispatch the model-specific prompt and generation path."""
    spec = prompt_for(model_id, task, shared_prompt)
    if model_id == PALIGEMMA_ID:
        answer, input_tokens, output_tokens, postprocess_info = generate_paligemma(
            model, processor, image, spec
        )
    elif model_id == MOLMO2_ID:
        answer, input_tokens, output_tokens, postprocess_info = generate_molmo2(
            model, processor, image, spec
        )
    elif model_id == FLORENCE_ID:
        answer, input_tokens, output_tokens, postprocess_info = generate_florence(
            model, processor, image, spec
        )
        if task in {"ai2d", "vqa_negation"}:
            answer, normalization = normalize_choice_text(answer, shared_prompt)
            postprocess_info["choice_normalization"] = normalization
    elif model_id == LLAMA_VISION_ID:
        answer, input_tokens, output_tokens, postprocess_info = generate_llama_vision(
            model, processor, image, spec
        )
    elif model_id in INTERNVL_GPU_IDS:
        if model_id in INTERNVL35_GGUF_CANDIDATES:
            answer, input_tokens, output_tokens, postprocess_info = model.generate(
                image, spec.text, max_new_tokens=256
            )
        else:
            answer, input_tokens, output_tokens, postprocess_info = generate_internvl(
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


def normalize_choice_text(answer: str, shared_prompt: str) -> tuple[str, str]:
    """Convert an exact multiple-choice option text to its existing label."""
    options = re.findall(r"(?m)^\s*([A-D])\)\s*(.+?)\s*$", shared_prompt)
    if not options:
        return answer, "not_applied_no_options"

    normalize = lambda value: re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)
    answer_key = normalize(answer)
    matches = [label for label, text in options if normalize(text) == answer_key]
    if len(matches) == 1:
        return matches[0], "exact_option_text_to_label"
    return answer, "not_applied_no_exact_match"
