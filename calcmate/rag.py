from __future__ import annotations

"""Format retrieved cases as few-shot reasoning examples (enhancement #1).

Retrieval no longer only feeds the deterministic CBR constraint layer: the
retrieved solved cases are rendered into the planning/fallback prompts so the
LLM reasons *with* concrete, structurally similar worked examples. This is the
"RAG done properly" part -- examples injected into the reasoning prompt, not
merely displayed alongside the answer.
"""

from calcmate.models import RetrievedCase


def format_retrieved_examples(cases: list[RetrievedCase], max_examples: int = 3) -> str:
    if not cases:
        return "(no similar solved cases retrieved)"

    blocks: list[str] = []
    for index, case in enumerate(cases[:max_examples], start=1):
        lines = [
            f"Example {index} (case_id={case.case_id}, similarity={case.score:.3f}):",
            f"  problem: {case.problem_text}",
            f"  knowns: {sorted(case.known_symbols)}",
            f"  unknown: {case.unknown}",
            f"  constraints_fired: {case.constraints_fired}",
            f"  equations_used: {case.equations_used}",
            f"  law_nodes: {case.law_nodes}",
        ]
        if case.reasoning_program:
            program = ", ".join(
                op.get("op", "?")
                + (f"({op.get('equation') or op.get('constraint') or op.get('law') or op.get('symbol') or ''})"
                   if op.get("op") != "verify" else "")
                for op in case.reasoning_program
            )
            lines.append(f"  reasoning_program: {program}")
        if case.solution_steps:
            steps = "; ".join(str(step) for step in case.solution_steps)
            lines.append(f"  solution_steps: {steps}")
        if case.final_answer:
            lines.append(f"  final_answer: {case.final_answer}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
