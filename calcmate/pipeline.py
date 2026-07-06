from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Protocol

from calcmate.extraction import DSPyPhysicsExtractor
from calcmate.knowledge_graph import load_default_graph
from calcmate.models import AttemptLog, ExtractedProblem, PhaseTrace, Solution, SolutionStep, UnitValidation
from calcmate.narration import Narrator
from calcmate.overlay import load_overlay
from calcmate.postgres_logging import AttemptLogger, NoopAttemptLogger
from calcmate.reasoning import PhysicsReasoner, ReasoningResult
from calcmate.retrieval import CaseRetriever, FaissCaseRetriever, InMemoryCaseRetriever


class Extractor(Protocol):
    def extract(self, text: str) -> ExtractedProblem:
        ...


class CalcMatePipeline:
    """Nine-phase CalcMate workflow.

    The LLM-facing phases are extraction and narration. Retrieval informs
    constraint handling, while graph traversal, SymPy, unit validation, and
    logging remain deterministic system components.
    """

    def __init__(
        self,
        extractor: Extractor | None = None,
        retriever: CaseRetriever | None = None,
        narrator: Narrator | None = None,
        attempt_logger: AttemptLogger | None = None,
    ) -> None:
        graph = load_default_graph()
        self.extractor = extractor or DSPyPhysicsExtractor()
        self.retriever = retriever or self._default_retriever()
        self.reasoner = PhysicsReasoner(graph)
        self.narrator = narrator or Narrator()
        self.attempt_logger = attempt_logger or NoopAttemptLogger()

    def _default_retriever(self) -> CaseRetriever:
        backend = os.environ.get("CALCMATE_RETRIEVAL_BACKEND", "memory").lower()
        if backend == "faiss":
            try:
                return FaissCaseRetriever()
            except Exception:
                return InMemoryCaseRetriever()
        return InMemoryCaseRetriever()

    def solve(self, text: str, overlay_id: str = "ncert") -> Solution:
        overlay = load_overlay(overlay_id)#loading the overlay
        phase_trace: list[PhaseTrace] = []#this list is used to track which values you got in which phase?

        extracted = self._phase_1_extract(text, phase_trace)#return ExtractedProblem(raw_text=text,quantities=quantities,target=target,trigger_phrases=triggers,domain_hint=domain_hint,)  this is wat is store inside extracted and phase list contains a few details of this extraction
        retrieved_cases = self._phase_2_retrieve(extracted, phase_trace)# this is wat the retrieved cases contain RetrievedCase(case_id=case.case_id,problem_text=case.problem_text,known_symbols=case.known_symbols,unknown=case.unknown,domain=case.domain,constraints_fired=case.constraints_fired,implied_values=case.implied_values,equations_used=case.equations_used,law_nodes=case.law_nodes,score=float(score),)
        reasoning = self.reasoner.solve(extracted, overlay, retrieved_cases) ##return ReasoningResult(problem,constraints_fired,steps,law_nodesunit_validation,phase_trace,)
        reasoning = self._apply_requested_output_unit(reasoning)
        phase_trace.extend(reasoning.phase_trace)# a copy of the phase trace
        narration = self._phase_8_narrate(reasoning, overlay, phase_trace)#explanation for answer has been fetched from llm or fallback method
        solution = self._build_solution(reasoning, overlay.overlay_id, retrieved_cases, narration, phase_trace)#jst a class to structure the answer
        self._phase_9_log(solution, phase_trace)#the final log 
        return solution

    def _phase_1_extract(self, text: str, trace: list[PhaseTrace]) -> ExtractedProblem:
        extracted = self.extractor.extract(text) #return ExtractedProblem(raw_text=text,quantities=quantities,target=target,trigger_phrases=triggers,domain_hint=domain_hint,)  this is wat is store inside extracted
        print("\n========== EXTRACTED ==========")
        print(extracted)
        print("===============================\n")
        trace.append(
            PhaseTrace(
                phase="1_extraction",
                status="ok",
                detail=f"Extracted knowns {sorted(extracted.quantities)} and target {extracted.target}.",
            )
        )
        return extracted

    def _phase_2_retrieve(self, extracted: ExtractedProblem, trace: list[PhaseTrace]):#gets similar embeddings/matches by domain matching or some other parameter
        cases = self.retriever.retrieve(extracted, top_k=3)

        print("\n===== RETRIEVED CASES =====")
        for case in cases:
          print(case)
        print("===========================\n")
        
        trace.append(
            PhaseTrace(
                phase="2_hybrid_retrieval",
                status="ok",
                detail=f"Retrieved {len(cases)} similar case(s).",
            )
        )
        return cases#contains the top 3-5 matches

    def _apply_requested_output_unit(self, reasoning: ReasoningResult) -> ReasoningResult:
        requested_unit = self._requested_output_unit(reasoning.problem.raw_text)
        if requested_unit is None:
            return reasoning

        final_step = reasoning.steps[-1]
        converted_step = self._convert_step_unit(final_step, requested_unit)
        if converted_step is None:
            trace = list(reasoning.phase_trace)
            trace.append(
                PhaseTrace(
                    phase="7b_output_unit_conversion",
                    status="skipped",
                    detail=f"Requested {requested_unit}, but no conversion is registered from {final_step.unit}.",
                )
            )
            return replace(reasoning, phase_trace=trace)

        steps = list(reasoning.steps)
        steps[-1] = converted_step
        trace = list(reasoning.phase_trace)
        trace.append(
            PhaseTrace(
                phase="7b_output_unit_conversion",
                status="ok",
                detail=f"Converted final answer from {final_step.unit} to {requested_unit}.",
            )
        )
        unit_validation = UnitValidation(
            expected_unit=reasoning.unit_validation.expected_unit,
            actual_unit=converted_step.unit,
            is_valid=True,
            message=f"Final answer converted from {final_step.unit} to requested unit {requested_unit}.",
        )
        return replace(reasoning, steps=steps, unit_validation=unit_validation, phase_trace=trace)

    def _requested_output_unit(self, text: str) -> str | None:
        lowered = text.lower()
        patterns = [
            (r"\bin\s+(?:centimeters?|centimetres?|cm)\b", "cm"),
            (r"\bin\s+(?:kilometers?|kilometres?|km)\b", "km"),
            (r"\bin\s+(?:meters?|metres?)\b", "m"),
        ]
        for pattern, unit in patterns:
            if re.search(pattern, lowered):
                return unit
        return None

    def _convert_step_unit(self, step: SolutionStep, requested_unit: str) -> SolutionStep | None:
        length_conversions = {
            ("m", "cm"): 100.0,
            ("m", "km"): 0.001,
            ("cm", "m"): 0.01,
            ("km", "m"): 1000.0,
        }
        if step.unit == requested_unit:
            return step
        factor = length_conversions.get((step.unit, requested_unit))
        if factor is None:
            return None
        return replace(step, value=round(step.value * factor, 4), unit=requested_unit)

    def _phase_8_narrate(self, reasoning: ReasoningResult, overlay, trace: list[PhaseTrace]) -> str:
        narration = self.narrator.narrate(
            reasoning.problem,
            overlay,
            reasoning.constraints_fired,
            reasoning.steps,
        )
        trace.append(
            PhaseTrace(
                phase="8_narration",
                status="ok",
                detail="Generated narration from verified solution payload.",
            )
        )
        return narration

    def _phase_9_log(self, solution: Solution, trace: list[PhaseTrace]) -> None:
        self.attempt_logger.log_attempt(#attempt log
            AttemptLog(
                problem_text=solution.problem.raw_text,
                overlay_id=solution.overlay_id,
                law_nodes_used=solution.law_nodes,
                constraints_fired=solution.applied_constraints,
                was_under_constrained=solution.was_under_constrained,
                was_contradiction=solution.was_contradiction,
            )
        )
        trace.append(
            PhaseTrace(
                phase="9_output_and_logging",
                status="ok",
                detail="Prepared output and recorded attempt log.",
            )
        )

    def _build_solution(
        self,
        reasoning: ReasoningResult,
        overlay_id: str,
        retrieved_cases,
        narration: str,
        phase_trace: list[PhaseTrace],
    ) -> Solution:
        answer = reasoning.steps[-1]
        return Solution(
            problem=reasoning.problem,
            overlay_id=overlay_id,
            applied_constraints=reasoning.constraints_fired,
            steps=reasoning.steps,
            answer_symbol=answer.solved_symbol,
            answer_value=answer.value,
            answer_unit=answer.unit,
            law_nodes=reasoning.law_nodes,
            narration=narration,
            retrieved_cases=retrieved_cases,
            unit_validation=reasoning.unit_validation,
            phase_trace=phase_trace,
            was_under_constrained=reasoning.was_under_constrained,
            was_contradiction=reasoning.was_contradiction,
        )
