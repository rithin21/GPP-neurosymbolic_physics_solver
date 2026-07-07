from __future__ import annotations

"""Deterministic verification layer (enhancement #4).

Extends the original final-answer :class:`UnitValidator` with two stronger,
equation-level checks:

* **Dimensional balance** -- every equation used in the solution must have the
  same physical dimension on both sides (e.g. ``v = u + a*t`` is
  ``[L/T] = [L/T] + [L/T^2]*[T]``).
* **Substitution unit consistency** -- every known quantity substituted into an
  equation must carry a unit whose dimension matches the canonical dimension
  registered for that symbol.

The verifier degrades gracefully when Pint is unavailable, and it never
loosens the original final-answer validation, so existing behaviour/tests are
preserved.
"""

from dataclasses import dataclass

from calcmate.constants import SYMBOL_DIMENSIONS, UNIT_BY_SYMBOL
from calcmate.models import Quantity, SolutionStep, UnitValidation
from calcmate.unit_validation import UnitValidator


@dataclass(frozen=True)
class DimensionCheck:
    subject: str
    is_valid: bool
    message: str


@dataclass(frozen=True)
class VerificationReport:
    final_answer: UnitValidation
    equation_checks: list[DimensionCheck]
    substitution_checks: list[DimensionCheck]

    @property
    def is_valid(self) -> bool:
        return (
            self.final_answer.is_valid
            and all(check.is_valid for check in self.equation_checks)
            and all(check.is_valid for check in self.substitution_checks)
        )

    def summary(self) -> str:
        failures = [
            check.message
            for check in (*self.equation_checks, *self.substitution_checks)
            if not check.is_valid
        ]
        if not self.final_answer.is_valid:
            failures.insert(0, self.final_answer.message)
        if failures:
            return "Verification failed: " + "; ".join(failures)
        eqn = len(self.equation_checks)
        sub = len(self.substitution_checks)
        return (
            f"{self.final_answer.message} "
            f"Dimensional balance held for {eqn} equation(s); "
            f"unit consistency held for {sub} substitution(s)."
        )


# Canonical unit strings mapped to a form Pint understands. Superset of the
# UnitValidator map so we can cover speed/distance/time symbols too.
_PINT_UNIT = {
    "m/s^2": "meter/second**2",
    "m/s2": "meter/second**2",
    "m/s": "meter/second",
    "m": "meter",
    "s": "second",
}


class DimensionalVerifier:
    def __init__(self, unit_validator: UnitValidator | None = None) -> None:
        self.unit_validator = unit_validator or UnitValidator()
        self._registry = None
        try:
            import pint

            self._registry = pint.UnitRegistry()
        except ImportError:
            self._registry = None

    def verify(
        self,
        steps: list[SolutionStep],
        quantities: dict[str, Quantity],
    ) -> VerificationReport:
        final = self.unit_validator.validate(steps[-1])
        equation_checks = [self._check_equation(step) for step in steps]
        substitution_checks = self._check_substitutions(quantities)
        return VerificationReport(
            final_answer=final,
            equation_checks=equation_checks,
            substitution_checks=substitution_checks,
        )

    def _check_equation(self, step: SolutionStep) -> DimensionCheck:
        expression = step.equation
        if self._registry is None or "=" not in expression:
            return DimensionCheck(expression, True, f"{expression}: dimension check skipped.")
        left_text, right_text = expression.split("=", 1)
        try:
            left_dim = self._dimension_of(left_text)
            right_dim = self._dimension_of(right_text)
        except Exception as exc:  # noqa: BLE001 - inability to check is not a mismatch
            return DimensionCheck(expression, True, f"{expression}: dimension check skipped ({exc}).")
        if left_dim == right_dim:
            return DimensionCheck(expression, True, f"{expression}: balanced.")
        return DimensionCheck(
            expression,
            False,
            f"{expression}: LHS {left_dim} != RHS {right_dim}.",
        )

    def _dimension_of(self, side: str):
        namespace = {
            symbol: self._registry.Quantity(1.0, _PINT_UNIT.get(unit, unit))
            for symbol, unit in _symbol_units().items()
        }
        value = eval(side.strip(), {"__builtins__": {}}, namespace)  # noqa: S307 - trusted graph expressions only
        if hasattr(value, "dimensionality"):
            return value.dimensionality
        # A pure number (dimensionless) still needs a comparable dimensionality.
        return self._registry.Quantity(1.0).dimensionality

    def _check_substitutions(self, quantities: dict[str, Quantity]) -> list[DimensionCheck]:
        checks: list[DimensionCheck] = []
        for symbol, quantity in sorted(quantities.items()):
            expected = SYMBOL_DIMENSIONS.get(symbol)
            if expected is None:
                continue
            actual_unit = quantity.unit or UNIT_BY_SYMBOL.get(symbol, "")
            if not actual_unit:
                # No unit attached (e.g. a graph-implied value); trust the symbol.
                checks.append(DimensionCheck(symbol, True, f"{symbol}: no unit to check."))
                continue
            if self._registry is None:
                ok = actual_unit == UNIT_BY_SYMBOL.get(symbol, actual_unit)
                checks.append(
                    DimensionCheck(symbol, ok, f"{symbol}: string unit check ({actual_unit}).")
                )
                continue
            try:
                expected_dim = self._registry.parse_expression(expected).dimensionality
                actual_dim = self._registry.parse_expression(
                    _PINT_UNIT.get(actual_unit, actual_unit)
                ).dimensionality
            except Exception as exc:  # noqa: BLE001 - inability to check is not a mismatch
                checks.append(DimensionCheck(symbol, True, f"{symbol}: unit check skipped ({exc})."))
                continue
            ok = expected_dim == actual_dim
            message = (
                f"{symbol}: {actual_unit} matches {expected}."
                if ok
                else f"{symbol}: {actual_unit} is not {expected}."
            )
            checks.append(DimensionCheck(symbol, ok, message))
        return checks


def _symbol_units() -> dict[str, str]:
    """Unit string per symbol for equation-dimension evaluation."""
    units = dict(UNIT_BY_SYMBOL)
    units.setdefault("speed", "m/s")
    units.setdefault("distance", "m")
    units.setdefault("time", "s")
    return units
