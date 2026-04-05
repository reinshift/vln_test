import re
import uuid
from typing import List

from vln_core.config.models import Mission, MissionStep


_CLAUSE_SPLIT = re.compile(
    r"(?:\s*,\s*|\s*;\s*|\b(?:and then|then|afterwards|after that|next)\b)",
    re.IGNORECASE,
)

_TARGET_PATTERN = re.compile(
    r"(?:toward|towards|to|near|into|through|past|beside|around)\s+(?:the\s+)?(?P<label>[a-z0-9][a-z0-9 \-_]+)",
    re.IGNORECASE,
)


def split_instruction(instruction: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", (instruction or "").strip())
    if not cleaned:
        return []
    return [part.strip(" .,!?:;") for part in _CLAUSE_SPLIT.split(cleaned) if part.strip(" .,!?:;")]


def _normalize_target(text: str) -> str:
    label = re.sub(r"\s+", " ", (text or "").strip().lower())
    label = re.sub(r"^(the|a|an)\s+", "", label)
    return label


def _action_for_clause(clause: str) -> str:
    lowered = clause.lower()
    if "right" in lowered or "veer right" in lowered:
        return "right"
    if "left" in lowered or "veer left" in lowered:
        return "left"
    if "back" in lowered or "reverse" in lowered:
        return "backward"
    return "forward"


def _target_for_clause(clause: str) -> str:
    match = _TARGET_PATTERN.search(clause)
    if match:
        return _normalize_target(match.group("label"))
    fallback = re.search(r"(?:the|a|an)\s+([a-z0-9][a-z0-9 \-_]+)$", clause, flags=re.IGNORECASE)
    if fallback:
        candidate = _normalize_target(fallback.group(1))
        if candidate not in {"left", "right", "front", "forward"}:
            return candidate
    return ""


def compile_instruction(instruction: str, mission_id: str = "") -> Mission:
    steps = []
    clauses = split_instruction(instruction)
    for index, clause in enumerate(clauses, start=1):
        steps.append(
            MissionStep(
                step_index=index,
                action=_action_for_clause(clause),
                target_label=_target_for_clause(clause),
                desired_distance_m=2.0,
                terminal=index == len(clauses),
            )
        )
    if not steps:
        steps = [MissionStep(step_index=1, action="forward", terminal=True)]
    return Mission(
        mission_id=mission_id or f"mission-{uuid.uuid4().hex[:8]}",
        raw_instruction=(instruction or "").strip(),
        steps=steps,
    )

