UNIT_BY_SYMBOL = {
    "u": "m/s",
    "v": "m/s",
    "a": "m/s^2",
    "t": "s",
    "s": "m",
}

# Pint-parseable canonical dimension for every symbol that can appear in a
# graph equation. Used by the DimensionalVerifier to check that each equation
# is dimensionally balanced (LHS dimension == RHS dimension) and that every
# substituted quantity carries a dimensionally consistent unit.
SYMBOL_DIMENSIONS = {
    "u": "meter/second",
    "v": "meter/second",
    "a": "meter/second**2",
    "t": "second",
    "s": "meter",
    "speed": "meter/second",
    "distance": "meter",
    "time": "second",
}
CANONICAL_SYMBOLS = {
    "u",
    "v",
    "a",
    "t",
    "s",
}

SYMBOL_ALIASES = {
    "u": {
        "u", "initial_velocity", "initial velocity",
        "vi", "v_i",
    },
    "v": {
        "v", "velocity", "speed",
        "final_velocity", "final velocity",
        "vf", "v_f",
        "avg speed", "average speed",
    },
    "a": {
        "a", "acceleration",
    },
    "t": {
        "t", "time",
        "elapsed time",
        "travel time",
        "duration",
    },
    "s": {
        "s",
        "d",
        "distance",
        "distance travelled",
        "distance traveled",
        "total distance",
        "displacement",
        "how far",
    },
}
