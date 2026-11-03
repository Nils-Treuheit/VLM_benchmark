#!/usr/bin/env python3
"""Download and verify the Llama 3.2 Vision 11B BNB 4-bit candidate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ID = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
DEFAULT_ROOT = Path(
    "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/llama3_2_vision_candidates"
)
FILENAME = "Llama-3.2-11B-Vision-Instruct-bnb-4bit"


def verify(snapshot: Path) -> dict:
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise RuntimeError("snapshot is missing config.json or model.safetensors.index.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    architecture = config.get("architectures", [])
    if "MllamaForConditionalGeneration" not in architecture:
        raise RuntimeError(f"unexpected Llama Vision architecture: {architecture}")
    shards = sorted(set(index.get("weight_map", {}).values()))
    missing = [name for name in shards if not (snapshot / name).is_file()]
    if not shards or missing:
        raise RuntimeError(f"incomplete checkpoint; missing shards: {missing or '(none indexed)'}")
    weight_bytes = sum((snapshot / name).stat().st_size for name in shards)
    if weight_bytes < 1_000_000_000:
        raise RuntimeError(f"checkpoint weights unexpectedly small: {weight_bytes} bytes")
    required = ("preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json")
    missing_support = [name for name in required if not (snapshot / name).is_file()]
    if missing_support:
        raise RuntimeError("snapshot is missing processor/tokenizer files: " + ", ".join(missing_support))
    return {
        "repo_id": REPO_ID,
        "architecture": architecture,
        "shards": shards,
        "weight_bytes": weight_bytes,
        "license": config.get("license", "llama3.2"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--revision", help="pin a known Hugging Face commit SHA")
    parser.add_argument("--check-only", action="store_true", help="verify existing files without downloading")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    snapshot = root / FILENAME
    if not args.check_only:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi()
        revision = args.revision or api.model_info(REPO_ID).sha
        if not revision:
            raise RuntimeError(f"could not resolve a pinned revision for {REPO_ID}")
        root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=REPO_ID,
            revision=revision,
            local_dir=str(snapshot),
            token=os.environ.get("HF_TOKEN"),
        )
    manifest = verify(snapshot)
    manifest["revision"] = args.revision
    manifest_path = root / "download_manifest.json"
    if not args.check_only:
        # The revision is re-read from the repository only when downloading.
        from huggingface_hub import HfApi

        manifest["revision"] = args.revision or HfApi().model_info(REPO_ID).sha
        manifest["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["user_license_scope_confirmation"] = (
            "User confirmed authorization for non-commercial academic local functional testing."
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elif manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["revision"] = saved.get("revision")
    print(json.dumps({**manifest, "snapshot": str(snapshot)}, indent=2))
    print("Llama 3.2 Vision files verified; no inference was run.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Llama 3.2 Vision preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Downloads are resumable; fix the reported issue and rerun.", file=sys.stderr)
        raise SystemExit(1)
