from calcmate.text_patterns import compile_phrase_set

# SUVAT symbols (u, v, a, t, s) plus the plain grade 6/7 quantities
# (speed, distance, time) that the graph also models as standalone nodes,
# plus average-velocity, vertical-projectile-time-of-flight, and two-body
# relative-motion symbols (see data/kinematics_graph.json: eq_avg_velocity,
# eq_time_of_flight, eq_relative_speed, eq_meeting_time).
UNIT_BY_SYMBOL = {
    "u": "m/s",
    "v": "m/s",
    "a": "m/s^2",
    "t": "s",
    "s": "m",
    "speed": "m/s",
    "distance": "m",
    "time": "s",
    "avg_v": "m/s",
    "v1": "m/s",
    "v2": "m/s",
    "relative_speed": "m/s",
    "direction": "",
    "separation": "m",
}
CANONICAL_SYMBOLS = set(UNIT_BY_SYMBOL)

# Base accepted spellings for each canonical symbol. These no longer need to
# enumerate every surface variant (plural, hyphenation, verb tense) - they
# are compiled into tolerant regex patterns below via SYMBOL_ALIAS_PATTERNS.
SYMBOL_ALIASES: dict[str, set[str]] = {
    "u": {
        "initial_velocity", "initial velocity",
        "vi", "v_i",
    },
    "v": {
        "velocity",
        "final_velocity", "final velocity",
        "vf", "v_f",
    },
    "a": {
        "acceleration",
    },
    "t": {
        "elapsed time",
        "travel time",
        "duration",
    },
    "s": {
        "distance travelled",
        "distance traveled",
        "total distance",
        "displacement",
    },
    "speed": {
        "avg speed", "average speed",
    },
    "distance": {
        "d",
    },
    "time": set(),
    "avg_v": {
        "average velocity", "mean velocity",
    },
    "v1": {
        "velocity of the first object", "first object's velocity",
        "speed of the first object", "v_1", "va",
    },
    "v2": {
        "velocity of the second object", "second object's velocity",
        "speed of the second object", "v_2", "vb",
    },
    "relative_speed": {
        "relative velocity", "relative speed",
    },
    "direction": set(),
    "separation": {
        "distance apart", "initial distance apart", "initial separation",
        "gap between them", "distance between them",
    },
}

# One compiled regex per canonical symbol, matching the symbol itself plus
# any of its aliases (with spacing/inflection tolerance). Used instead of
# exact set-membership checks so phrasing variants aren't silently missed.
SYMBOL_ALIAS_PATTERNS: dict[str, "re.Pattern[str]"] = {
    canonical: compile_phrase_set(aliases | {canonical})
    for canonical, aliases in SYMBOL_ALIASES.items()
}
