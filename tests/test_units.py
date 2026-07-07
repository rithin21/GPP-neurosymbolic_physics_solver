import unittest

from calcmate.models import ExtractedProblem, Quantity
from calcmate.pipeline import CalcMatePipeline
from calcmate.units import convert, parse_requested_unit, to_si


class UnitsModuleTests(unittest.TestCase):
    def test_to_si_converts_to_symbol_canonical_unit(self):
        self.assertEqual(to_si(36, "km", "s"), (36000.0, "m"))
        self.assertEqual(to_si(0.75, "h", "t"), (2700.0, "s"))
        # Already-SI inputs are unchanged (identity).
        self.assertEqual(to_si(20, "m/s", "u"), (20.0, "m/s"))

    def test_convert_speed_units(self):
        self.assertAlmostEqual(convert(13.3333333, "m/s", "km/h"), 48.0, places=3)
        self.assertAlmostEqual(convert(1.0, "m/s", "km/min"), 0.06, places=6)

    def test_parse_requested_unit(self):
        self.assertEqual(parse_requested_unit("average speed in km/h?"), "km/h")
        self.assertEqual(parse_requested_unit("speed in kilometers per minute?"), "km/min")
        self.assertEqual(parse_requested_unit("Find the maximum height."), None)


class _BusExtractor:
    """Mimics extraction of the bus problem: distance/time aliased to s/t,
    stated in km and hours."""

    def extract(self, text: str) -> ExtractedProblem:
        return ExtractedProblem(
            raw_text=text,
            target="v",
            trigger_phrases=["travels"],
            quantities={
                "s": Quantity("s", 36, "km", "36 km"),
                "t": Quantity("t", 0.75, "h", "0.75 hours"),
            },
        )


class _MayaExtractor:
    def extract(self, text: str) -> ExtractedProblem:
        return ExtractedProblem(
            raw_text=text,
            target="v",
            trigger_phrases=["walks"],
            quantities={
                "s": Quantity("s", 1.2, "km", "1.2 km"),
                "t": Quantity("t", 20, "min", "20 minutes"),
            },
        )


class UnitAwarePipelineTests(unittest.TestCase):
    def test_bus_average_speed_in_kmph(self):
        pipeline = CalcMatePipeline(extractor=_BusExtractor())
        solution = pipeline.solve(
            "A school bus travels 36 km in 0.75 hours. What is the average speed in km/h?",
            "ncert",
        )
        self.assertEqual(solution.answer_value, 48.0)
        self.assertEqual(solution.answer_unit, "km/h")
        self.assertTrue(solution.unit_validation.is_valid)
        phases = {t.phase for t in solution.phase_trace}
        self.assertIn("1b_unit_normalization", phases)

    def test_maya_speed_in_km_per_minute(self):
        pipeline = CalcMatePipeline(extractor=_MayaExtractor())
        solution = pipeline.solve(
            "Maya walks 1.2 km in 20 minutes. What is her average speed in kilometers per minute?",
            "ncert",
        )
        self.assertEqual(solution.answer_value, 0.06)
        self.assertEqual(solution.answer_unit, "km/min")


if __name__ == "__main__":
    unittest.main()
