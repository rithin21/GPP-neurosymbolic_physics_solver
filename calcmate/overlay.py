from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from calcmate.knowledge_graph import PROJECT_ROOT


@dataclass(frozen=True)
class Overlay:
    overlay_id: str
    name: str
    equation_priority: list[str]
    blocked_methods: list[str]
    show_intermediate_steps: bool
    show_units: bool
    narration_tone: str
    structure: str


def load_overlay(overlay_id: str) -> Overlay:
    path = PROJECT_ROOT / "data" / "overlays" / f"{overlay_id}_kinematics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return Overlay(
        overlay_id=data["id"],
        name=data["name"],
        equation_priority=data["equation_priority"],
        blocked_methods=data.get("blocked_methods", []),
        show_intermediate_steps=data.get("show_intermediate_steps", True),
        show_units=data.get("show_units", True),
        narration_tone=data.get("narration_tone", "clear"),
        structure=data.get("structure", "solution"),
    )
