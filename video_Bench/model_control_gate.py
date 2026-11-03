#!/usr/bin/env python3
"""Read-only check that managed local model services and GPU inventory are idle."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_SRC = Path("/home/nils/Project_Code/codex_opencode_pipeline/src")
sys.path.insert(0, str(PIPELINE_SRC))

from executor import _control_state_is_released  # noqa: E402
from opencode_runtime import model_control_request  # noqa: E402


def main() -> int:
    try:
        status_code, state = model_control_request("/status")
        models_code, _ = model_control_request("/models")
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"Model-control read failed; refusing to proceed: {error}", file=sys.stderr)
        return 3

    if status_code != 200 or models_code != 200:
        print(
            f"Model-control status unavailable (status={status_code}, models={models_code}); refusing to proceed.",
            file=sys.stderr,
        )
        return 3
    if not _control_state_is_released(state):
        print("A local model or GPU process is active; refusing release and benchmark launch.", file=sys.stderr)
        return 4
    print("Model-control /status and /models are reachable; managed engines and GPU inventory are idle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
