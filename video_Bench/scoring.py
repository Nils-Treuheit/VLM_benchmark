"""Shared deterministic parsers for the video benchmark outputs."""

from __future__ import annotations

import json
import math
import re
from typing import Any

INTERVAL_PARSER_VERSION = "video-vlm-intervals-v2"


def parse_choice(text: str | None, choices: list[str]) -> tuple[str | None, str]:
    if not text:
        return None, "empty_output"
    match = re.search(r"(?:^|\b)([A-E])(?:\b|[).:]|$)", text.strip(), re.IGNORECASE)
    if match:
        index = ord(match.group(1).upper()) - 65
        if index < len(choices):
            return choices[index], "letter"
    normalized = " ".join(text.casefold().split())
    exact = [choice for choice in choices if " ".join(choice.casefold().split()) == normalized]
    if len(exact) == 1:
        return exact[0], "exact_choice_text"
    return None, "unparsed_choice"


def parse_intervals(text: str | None) -> tuple[list[list[float]], str | None, str]:
    if not text:
        return [], "empty_output", "invalid"
    content = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1).strip()
    left = content.find("[")
    right = content.rfind("]")
    if left < 0 or right < left:
        return [], "no_json_array", "invalid"
    try:
        value: Any = json.loads(content[left:right + 1])
        if not isinstance(value, list):
            raise ValueError("interval output is not an array")
        parse_status = "numeric_values"
        if (len(value) == 2 and all(not isinstance(part, (list, tuple, dict)) for part in value)):
            value = [value]
            parse_status = "single_interval_pair"
        result: list[list[float]] = []
        saw_numeric_string = False
        for interval in value:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError("each interval must contain two values")
            numeric = []
            for number in interval:
                if isinstance(number, bool) or not isinstance(number, (int, float, str)):
                    raise ValueError("interval endpoints must be numeric")
                if isinstance(number, str):
                    saw_numeric_string = True
                    if not number.strip():
                        raise ValueError("interval endpoints must be numeric")
                value_number = float(number)
                if not math.isfinite(value_number):
                    raise ValueError("interval endpoints must be finite")
                numeric.append(value_number)
            if numeric[0] > numeric[1]:
                raise ValueError("interval start exceeds end")
            result.append(numeric)
        if saw_numeric_string:
            parse_status = "numeric_strings"
        elif parse_status != "single_interval_pair":
            parse_status = "numeric_values"
        return result, None, parse_status
    except (json.JSONDecodeError, ValueError, TypeError, OverflowError) as error:
        return [], f"invalid_interval_json: {error}", "invalid"
