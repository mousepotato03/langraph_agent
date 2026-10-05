"""Validation shared by model-output consumers."""

import json
import re

from core.config import MAX_SUBTASKS


def extract_json(text: str) -> str:
    """Extract the first complete JSON object/array, including fenced responses."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return text[match.start() : match.start() + end]
    raise ValueError("응답에 유효한 JSON 객체 또는 배열이 없습니다.")


def validate_subtasks(value: object) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_SUBTASKS:
        raise ValueError(f"계획은 1~{MAX_SUBTASKS}개의 작업이어야 합니다.")
    tasks = []
    for task in value:
        description = task.get("description") if isinstance(task, dict) else task
        if not isinstance(description, str) or not description.strip():
            raise ValueError("각 작업에는 비어 있지 않은 설명이 필요합니다.")
        tasks.append(description.strip())
    return tasks


def merge_preferences(existing: dict | None, new: object) -> dict:
    """Store only the profile schema; keep list order stable across updates."""
    if not isinstance(new, dict):
        raise ValueError("선호도는 JSON 객체여야 합니다.")
    merged = dict(existing or {})
    for key in ("preferred_categories", "interests"):
        values = new.get(key, [])
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ValueError(f"{key}는 문자열 배열이어야 합니다.")
        merged[key] = list(dict.fromkeys([*merged.get(key, []), *values]))
    for key in ("price_preference", "skill_level", "notes"):
        value = new.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"{key}는 문자열이어야 합니다.")
        if value.strip():
            merged[key] = value.strip()
    return merged
