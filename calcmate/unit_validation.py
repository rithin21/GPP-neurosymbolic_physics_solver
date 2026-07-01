from __future__ import annotations

from calcmate.constants import UNIT_BY_SYMBOL #ground truth table for units of specific quantities
from calcmate.models import SolutionStep, UnitValidation


class UnitValidator:
    def __init__(self) -> None:#try to use pint if the library is installed
        self._registry = None
        try:
            import pint

            self._registry = pint.UnitRegistry()
        except ImportError:
            self._registry = None

    def validate(self, step: SolutionStep) -> UnitValidation:
        expected = UNIT_BY_SYMBOL.get(step.solved_symbol, "")#get the expected unit from the lookup table
        actual = step.unit #watever the solver attached to the step
        if not expected:
            return UnitValidation(expected, actual, True, "No expected unit registered.")
        if self._registry is None:
            return UnitValidation(expected, actual, expected == actual, "Pint not installed; used string unit check.")

        try:#converts the units to a form that pint can understand 
            expected_unit = self._registry.parse_expression(self._to_pint(expected))
            actual_unit = self._registry.parse_expression(self._to_pint(actual))
        except Exception as exc:
            return UnitValidation(expected, actual, False, f"Unit parse failed: {exc}")

        is_valid = expected_unit.dimensionality == actual_unit.dimensionality#checks the dimensionality
        message = "Unit dimensions match." if is_valid else "Unit dimensions do not match."
        return UnitValidation(expected, actual, is_valid, message)

    def _to_pint(self, unit: str) -> str:#the fn that is used to convert into pint friendly formats
        unit_map = {
            "m/s^2": "meter/second**2",
            "m/s": "meter/second",
            "m": "meter",
            "s": "second",
        }
        return unit_map.get(unit, unit)
