from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4


@dataclass
class Attempt:
    id: str
    problem_text: str
    overlay_id: str
    law_nodes: list[str]
    constraints_fired: list[str]
    was_under_constrained: bool
    was_contradiction: bool
    correct: bool
    created_at: str


@dataclass
class AttemptStore:
    """Tiny runtime store with the same shape as the PostgreSQL schema."""

    attempts: list[Attempt] = field(default_factory=list)

    def record(
        self,
        problem_text: str,
        overlay_id: str,
        law_nodes: list[str],
        constraints_fired: list[str] | None = None,
        was_under_constrained: bool = False,
        was_contradiction: bool = False,
        correct: bool = True,
    ) -> Attempt:
        attempt = Attempt(
            id=str(uuid4()),
            problem_text=problem_text,
            overlay_id=overlay_id,
            law_nodes=law_nodes,
            constraints_fired=constraints_fired or [],
            was_under_constrained=was_under_constrained,
            was_contradiction=was_contradiction,
            correct=correct,
            created_at=datetime.now(UTC).isoformat(),
        )
        self.attempts.append(attempt)
        return attempt

    def weak_nodes(self) -> list[dict[str, int | str]]:
        wrong_counts: Counter[str] = Counter()
        seen_counts: Counter[str] = Counter()
        for attempt in self.attempts:
            for node in attempt.law_nodes:
                seen_counts[node] += 1
                if not attempt.correct:
                    wrong_counts[node] += 1
        return [
            {"law_node": node, "attempts": seen_counts[node], "mistakes": wrong_counts[node]}
            for node in sorted(seen_counts, key=lambda item: (-wrong_counts[item], item))
        ]
