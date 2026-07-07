from __future__ import annotations

"""Turn the knowledge graph into *reasoning context* (enhancement #2).

The graph is no longer only an equation/constraint lookup table: this module
projects the portion of the graph relevant to a problem into a compact,
prompt-ready description of the laws, equations, and constraints available,
annotated with how they relate to what is currently known. That description is
fed to the planning and fallback LLM stages so the model reasons *with* the
graph instead of from scratch.
"""

from dataclasses import dataclass, field

from calcmate.knowledge_graph import PhysicsKnowledgeGraph
from calcmate.models import ExtractedProblem


@dataclass(frozen=True)
class EquationView:
    node_id: str
    expression: str
    symbols: list[str]
    law: str
    solvable_now_for: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConstraintView:
    node_id: str
    name: str
    trigger_phrases: list[str]
    implies: dict[str, float]
    applies_to_equations: list[str]


@dataclass(frozen=True)
class GraphReasoningContext:
    chapter: str
    equations: list[EquationView]
    constraints: list[ConstraintView]
    laws: list[str]

    def equation_ids(self) -> list[str]:
        return [equation.node_id for equation in self.equations]

    def constraint_ids(self) -> list[str]:
        return [constraint.node_id for constraint in self.constraints]

    def to_prompt_text(self) -> str:
        lines = [f"Chapter: {self.chapter}", "", "Available equations (node_id: expression):"]
        for equation in self.equations:
            hint = (
                f"  [can be solved now for: {', '.join(equation.solvable_now_for)}]"
                if equation.solvable_now_for
                else ""
            )
            lines.append(f"  - {equation.node_id}: {equation.expression}  (law {equation.law}){hint}")
        lines.append("")
        lines.append("Applicable constraints (node_id -> implied values, triggers):")
        for constraint in self.constraints:
            implies = ", ".join(f"{sym}={val}" for sym, val in constraint.implies.items()) or "(none)"
            triggers = ", ".join(constraint.trigger_phrases) or "(none)"
            lines.append(f"  - {constraint.node_id}: implies {implies}; triggers: {triggers}")
        return "\n".join(lines)


def build_reasoning_context(
    graph: PhysicsKnowledgeGraph,
    problem: ExtractedProblem,
) -> GraphReasoningContext:
    known = set(problem.quantities)

    equations: list[EquationView] = []
    for node_id, attrs in graph.nodes_by_type("equation"):
        symbols = list(attrs.get("symbols", []))
        law_nodes = graph.incoming(node_id, "expressed_as")
        unknowns = set(symbols) - known
        solvable_now = sorted(unknowns) if len(unknowns) == 1 else []
        equations.append(
            EquationView(
                node_id=node_id,
                expression=attrs.get("expression", ""),
                symbols=symbols,
                law=law_nodes[0] if law_nodes else "law_unknown",
                solvable_now_for=solvable_now,
            )
        )

    constraints: list[ConstraintView] = []
    for node_id, attrs in graph.nodes_by_type("constraint"):
        constraints.append(
            ConstraintView(
                node_id=node_id,
                name=attrs.get("name", node_id),
                trigger_phrases=list(attrs.get("trigger_phrases", [])),
                implies=dict(attrs.get("implies", {})),
                applies_to_equations=graph.incoming(node_id, "applicable_when"),
            )
        )

    laws = [name for _, attrs in graph.nodes_by_type("law") for name in [attrs.get("name", "")] if name]

    return GraphReasoningContext(
        chapter=graph.meta.get("chapter", ""),
        equations=equations,
        constraints=constraints,
        laws=laws,
    )
