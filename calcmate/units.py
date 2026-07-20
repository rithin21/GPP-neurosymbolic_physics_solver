from __future__ import annotations

"""SI-first unit conversion.

Every quantity in this chapter is solved in SI (m, s, m/s, m/s^2) - see
calcmate.constants.UNIT_BY_SYMBOL. This module only handles the last-mile
step: converting a solved SI value into whatever unit the question actually
asked for (e.g. "find the speed in km/h").

Tries Pint first (general unit algebra) and falls back to a small manual
conversion table when Pint isn't installed, mirroring the fallback pattern
already used in calcmate.unit_validation.UnitValidator.
"""

SI_UNIT_BY_KIND = {
    "length": "m",
    "time": "s",
    "speed": "m/s",
    "acceleration": "m/s^2",
}

# Conversion factor: 1 unit of X equals FACTOR SI units of the same kind.
_LENGTH_TO_M = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
}
_TIME_TO_S = {
    "ms": 0.001,
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
}
_SPEED_TO_MPS = {
    "m/s": 1.0,
    "km/h": 1000.0 / 3600.0,
    "km/min": 1000.0 / 60.0,
    "cm/s": 0.01,
}
_ACCELERATION_TO_MPS2 = {
    "m/s^2": 1.0,
    "km/h^2": 1000.0 / (3600.0**2),
}

_UNIT_TABLES = {
    "length": _LENGTH_TO_M,
    "time": _TIME_TO_S,
    "speed": _SPEED_TO_MPS,
    "acceleration": _ACCELERATION_TO_MPS2,
}


def unit_kind(unit: str) -> str | None:
    for kind, table in _UNIT_TABLES.items():
        if unit in table:
            return kind
    return None


def _convert_manual(value: float, from_unit: str, to_unit: str) -> float | None:
    kind = unit_kind(from_unit)
    if kind is None or to_unit not in _UNIT_TABLES[kind]:
        return None
    table = _UNIT_TABLES[kind]
    value_in_si = value * table[from_unit]
    return value_in_si / table[to_unit]


_PINT_UNIT_ALIASES = {
    "m/s^2": "meter/second**2",
    "km/h^2": "kilometer/hour**2",
    "m/s": "meter/second",
    "km/h": "kilometer/hour",
    "km/min": "kilometer/minute",
    "cm/s": "centimeter/second",
    "mm": "millimeter",
    "cm": "centimeter",
    "m": "meter",
    "km": "kilometer",
    "ms": "millisecond",
    "s": "second",
    "min": "minute",
    "h": "hour",
}


def _to_pint_expr(unit: str) -> str:
    return _PINT_UNIT_ALIASES.get(unit, unit)


def convert(value: float, from_unit: str, to_unit: str) -> float | None:
    """Convert an SI-solved value into the requested unit.

    Returns None if the units are unknown or of incompatible physical kind
    (the caller should leave the value in SI rather than guess).
    """
    if from_unit == to_unit:
        return value

    try:
        import pint  # type: ignore

        registry = pint.UnitRegistry()
        quantity = value * registry.parse_expression(_to_pint_expr(from_unit))
        converted = quantity.to(_to_pint_expr(to_unit))
        return float(converted.magnitude)
    except ImportError:
        pass
    except Exception:
        return None

    return _convert_manual(value, from_unit, to_unit)
