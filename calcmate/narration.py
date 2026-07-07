from __future__ import annotations

import json
import os

from calcmate.models import ExtractedProblem, SolutionStep
from calcmate.overlay import Overlay


def _build_narration_signature():#initializing dspy and telling it clearly wat the input and output should look like
    import dspy

    class NarrateVerifiedSolution(dspy.Signature):
        """Rewrite a verified physics solution for a student.

        You may only use the supplied verified_solution_json. State the given
        values, the equation used, and the final answer EXACTLY as provided.

        Hard rules:
        - Do NOT convert units. The answer's unit is already correct.
        - Do NOT recompute, re-derive, or double-check any number.
        - Do NOT question, dispute, or flag the answer as wrong; it is verified.
        - Do NOT introduce any number that is not in verified_solution_json.
        Match the requested institution style.
        """

        verified_solution_json: str = dspy.InputField()
        narration: str = dspy.OutputField(desc="Student-facing explanation only.")

    return NarrateVerifiedSolution


class Narrator:
    """Layer 3. DSPy narration over verified steps, with template fallback."""

    def __init__(self) -> None:
        self.narrate_program = None
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or os.environ.get("CALCMATE_USE_DSPY_NARRATOR", "1") == "0":
            return

        import dspy

        model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        api_base = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        lm = dspy.LM(
            f"openai/{model}",
            api_key=api_key,
            api_base=api_base,
            temperature=0.2,
            max_tokens=900,
        )
        dspy.configure(lm=lm)
        self.narrate_program = dspy.Predict(_build_narration_signature())

    def narrate(
        self,
        problem: ExtractedProblem,
        overlay: Overlay,
        constraints: list[str],
        steps: list[SolutionStep],
    ) -> str:
        payload = self._verified_payload(problem, overlay, constraints, steps)#This creates a structured JSON-like packet containing only verified information.
        if self.narrate_program is not None:#if dspy narrator is available
            prediction = self.narrate_program(verified_solution_json=json.dumps(payload))#answer from llm
            narration = getattr(prediction, "narration", "").strip()
            if narration and self._is_faithful(narration, steps):
                return narration
        return self._template_narration(problem, overlay, constraints, steps)#fallback method if llm is not connected (or the LLM narration was unfaithful)

    def _is_faithful(self, narration: str, steps: list[SolutionStep]) -> bool:
        """Reject LLM narration that disputes or diverges from the verified answer.

        Guards against the failure mode where the narrator re-derives the answer,
        invents a unit conversion, and then contradicts the verified value.
        """
        text = narration.lower()
        contradiction_cues = (
            "error",
            "discrepancy",
            "however",
            "seems",
            "incorrect",
            "mistake",
            "should be",
            "does not match",
            "doesn't match",
            "wrong",
            "actually",
        )
        if any(cue in text for cue in contradiction_cues):
            return False
        final = steps[-1]
        # The verified final value must appear (so the narration is about it).
        return f"{final.value:g}".lower() in text.replace(",", "")

    def _verified_payload(
        self,
        problem: ExtractedProblem,
        overlay: Overlay,
        constraints: list[str],
        steps: list[SolutionStep],
    ) -> dict:
        return {
            "institution_style": {
                "overlay_id": overlay.overlay_id,
                "tone": overlay.narration_tone,
                "structure": overlay.structure,
                "show_intermediate_steps": overlay.show_intermediate_steps,
                "show_units": overlay.show_units,
            },
            "given": {
                symbol: {"value": quantity.value, "unit": quantity.unit}
                for symbol, quantity in sorted(problem.quantities.items())
            },
            "target": problem.target,
            "constraints_fired": constraints,
            "verified_steps": [
                {
                    "equation": step.equation,
                    "substitution": step.substitution,
                    "solved_symbol": step.solved_symbol,
                    "value": step.value,
                    "unit": step.unit,
                }
                for step in steps
            ],
        }

    def _template_narration(#manual construction of answer from certain details that we have
        self,
        problem: ExtractedProblem,
        overlay: Overlay,
        constraints: list[str],
        steps: list[SolutionStep],
    ) -> str:
        step = steps[-1]
        unit = f" {step.unit}" if overlay.show_units and step.unit else ""
        lines: list[str] = []

        if overlay.structure == "given_find":
            given = ", ".join(
                f"{symbol} = {quantity.value:g} {quantity.unit}".strip()
                for symbol, quantity in sorted(problem.quantities.items())
                if symbol != step.solved_symbol
            )
            lines.append(f"Given: {given}")
            lines.append(f"Find: {step.solved_symbol}")

        if constraints and overlay.show_intermediate_steps:
            lines.append("Graph constraint used: " + ", ".join(constraints))

        if overlay.show_intermediate_steps:
            substituted = ", ".join(f"{key}={value:g}" for key, value in step.substitution.items())
            lines.append(f"Use {step.equation} with {substituted}.")
            lines.append(f"So, {step.solved_symbol} = {step.value:g}{unit}.")
        else:
            lines.append(f"{step.solved_symbol} = {step.value:g}{unit}.")

        return "\n".join(lines)

