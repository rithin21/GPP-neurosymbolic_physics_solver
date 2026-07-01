from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Quantity:
    symbol: str
    value: float
    unit: str
    source_text: str


@dataclass(frozen=True)
class ExtractedProblem:
    raw_text: str
    quantities: dict[str, Quantity]
    target: str
    trigger_phrases: list[str] = field(default_factory=list)
    domain_hint: str = "kinematics"

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "target": self.target,
            "domain_hint": self.domain_hint,
            "trigger_phrases": self.trigger_phrases,
            "quantities": {
                key: {
                    "symbol": quantity.symbol,
                    "value": quantity.value,
                    "unit": quantity.unit,
                    "source_text": quantity.source_text,
                }
                for key, quantity in self.quantities.items()
            },
        }


@dataclass(frozen=True)
class EquationCandidate:
    node_id: str
    expression: str
    unknown: str
    known_symbols: set[str]
    law_node: str


@dataclass(frozen=True)
class SolutionStep:
    equation_node: str
    law_node: str
    equation: str
    substitution: dict[str, float]
    solved_symbol: str
    value: float
    unit: str


@dataclass(frozen=True)
class RetrievedCase:
    case_id: str
    problem_text: str
    known_symbols: set[str]
    unknown: str
    domain: str
    constraints_fired: list[str]
    implied_values: dict[str, float]
    equations_used: list[str]
    law_nodes: list[str]
    score: float


@dataclass(frozen=True)
class UnitValidation:
    expected_unit: str
    actual_unit: str
    is_valid: bool
    message: str


@dataclass(frozen=True)
class PhaseTrace:
    phase: str
    status: str
    detail: str


@dataclass(frozen=True)
class AttemptLog:
    problem_text: str
    overlay_id: str
    law_nodes_used: list[str]
    constraints_fired: list[str]
    was_under_constrained: bool
    was_contradiction: bool


@dataclass(frozen=True)
class Solution:
    problem: ExtractedProblem
    overlay_id: str
    applied_constraints: list[str]
    steps: list[SolutionStep]
    answer_symbol: str
    answer_value: float
    answer_unit: str
    law_nodes: list[str]
    narration: str
    retrieved_cases: list[RetrievedCase] = field(default_factory=list)
    unit_validation: UnitValidation | None = None
    phase_trace: list[PhaseTrace] = field(default_factory=list)
    was_under_constrained: bool = False
    was_contradiction: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "overlay_id": self.overlay_id,
            "extraction": self.problem.to_jsonable(),
            "applied_constraints": self.applied_constraints,
            "steps": [
                {
                    "equation_node": step.equation_node,
                    "law_node": step.law_node,
                    "equation": step.equation,
                    "substitution": step.substitution,
                    "solved_symbol": step.solved_symbol,
                    "value": step.value,
                    "unit": step.unit,
                }
                for step in self.steps
            ],
            "answer": {
                "symbol": self.answer_symbol,
                "value": self.answer_value,
                "unit": self.answer_unit,
            },
            "law_nodes": self.law_nodes,
            "narration": self.narration,
            "retrieved_cases": [
                {
                    "case_id": case.case_id,
                    "score": case.score,
                    "constraints_fired": case.constraints_fired,
                    "equations_used": case.equations_used,
                    "law_nodes": case.law_nodes,
                }
                for case in self.retrieved_cases
            ],
            "unit_validation": None
            if self.unit_validation is None
            else {
                "expected_unit": self.unit_validation.expected_unit,
                "actual_unit": self.unit_validation.actual_unit,
                "is_valid": self.unit_validation.is_valid,
                "message": self.unit_validation.message,
            },
            "phase_trace": [
                {"phase": item.phase, "status": item.status, "detail": item.detail}
                for item in self.phase_trace
            ],
            "was_under_constrained": self.was_under_constrained,
            "was_contradiction": self.was_contradiction,
        }
