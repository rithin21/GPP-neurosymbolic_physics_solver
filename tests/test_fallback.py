import unittest

from calcmate.fallback import FallbackProposal
from calcmate.models import ExtractedProblem, Quantity
from calcmate.pipeline import CalcMatePipeline
from calcmate.retrieval import InMemoryCaseRetriever


def _empty_retriever() -> InMemoryCaseRetriever:
    # No retrieved cases -> the CBR constraint layer cannot fire, so the
    # problem stays under-constrained and the LLM fallback path is exercised.
    return InMemoryCaseRetriever(cases=[])


class UpwardThrowExtractor:
    """Extracts an upward-throw problem WITHOUT the trigger phrases, so the
    deterministic constraint layer cannot fire v=0 / a=-g on its own."""

    def extract(self, text: str) -> ExtractedProblem:
        return ExtractedProblem(
            raw_text=text,
            target="s",
            trigger_phrases=[],
            quantities={"u": Quantity("u", 20, "m/s", "launch speed 20 m/s")},
        )


class FakeFallback:
    def __init__(self, constraint_ids):
        self.constraint_ids = constraint_ids

    def attempt(self, problem, graph_context, retrieved_cases, failure_reason):
        return FallbackProposal(constraint_ids=self.constraint_ids, rationale="fake")


class FallbackTests(unittest.TestCase):
    PROBLEM = "A projectile leaves the launcher at 20 m/s. Find the peak rise."

    def test_fallback_resolves_under_constrained_via_graph_constraints(self):
        pipeline = CalcMatePipeline(
            extractor=UpwardThrowExtractor(),
            retriever=_empty_retriever(),
            fallback=FakeFallback(
                ["constraint_max_height_v_zero", "constraint_free_fall_upward_a_minus_g"]
            ),
        )
        solution = pipeline.solve(self.PROBLEM, "jee")

        self.assertFalse(solution.was_unresolved)
        self.assertEqual(solution.answer_symbol, "s")
        self.assertEqual(solution.answer_value, 20.4082)
        self.assertEqual(solution.steps[0].equation_node, "eq_v2_u2_2as")
        phases = {trace.phase: trace.status for trace in solution.phase_trace}
        self.assertEqual(phases["5b_llm_fallback"], "ok")

    def test_empty_fallback_yields_graceful_unresolved(self):
        pipeline = CalcMatePipeline(
            extractor=UpwardThrowExtractor(),
            retriever=_empty_retriever(),
            fallback=FakeFallback([]),
        )
        solution = pipeline.solve(self.PROBLEM, "jee")

        self.assertTrue(solution.was_unresolved)
        self.assertIsNone(solution.answer_value)
        self.assertEqual(solution.steps, [])
        self.assertIn("review", solution.narration.lower())
        phases = {trace.phase for trace in solution.phase_trace}
        self.assertIn("5c_unresolved", phases)

    def test_no_fallback_configured_still_raises(self):
        pipeline = CalcMatePipeline(extractor=UpwardThrowExtractor(), retriever=_empty_retriever())
        with self.assertRaises(ValueError):
            pipeline.solve(self.PROBLEM, "jee")


if __name__ == "__main__":
    unittest.main()
