import unittest

from calcmate.models import Quantity, SolutionStep
from calcmate.verification import DimensionalVerifier


def _step(equation: str, solved_symbol: str, unit: str) -> SolutionStep:
    return SolutionStep(
        equation_node="eq_test",
        law_node="law_test",
        equation=equation,
        substitution={},
        solved_symbol=solved_symbol,
        value=1.0,
        unit=unit,
    )


class DimensionalVerifierTests(unittest.TestCase):
    def test_balanced_equation_and_consistent_units_pass(self):
        verifier = DimensionalVerifier()
        steps = [_step("v = u + a*t", "v", "m/s")]
        quantities = {
            "u": Quantity("u", 0.0, "m/s", ""),
            "a": Quantity("a", 2.0, "m/s^2", ""),
            "t": Quantity("t", 5.0, "s", ""),
        }
        report = verifier.verify(steps, quantities)
        self.assertTrue(report.is_valid)
        self.assertTrue(all(check.is_valid for check in report.equation_checks))

    def test_dimensionally_broken_equation_fails(self):
        verifier = DimensionalVerifier()
        # s (length) cannot equal v (length/time).
        steps = [_step("s = v", "s", "m")]
        report = verifier.verify(steps, {"v": Quantity("v", 3.0, "m/s", "")})
        self.assertFalse(report.is_valid)
        self.assertIn("!=", report.summary())

    def test_inconsistent_substitution_unit_fails(self):
        verifier = DimensionalVerifier()
        steps = [_step("v = u + a*t", "v", "m/s")]
        # t is given a length unit -> dimensionally inconsistent for time.
        quantities = {"t": Quantity("t", 5.0, "m", "")}
        report = verifier.verify(steps, quantities)
        self.assertFalse(report.is_valid)


if __name__ == "__main__":
    unittest.main()
