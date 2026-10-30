"""Validate types/control, bound footprints, and assign storage/diagnostic IDs.

No scheduling optimization is performed: failure order is observable. Native
arithmetic intermediates must fit a signed 64-bit carrier by range analysis.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import lcm

from .ir import (
    BOOL,
    SCHEMA,
    TIME,
    Bits,
    Cancel,
    Cell,
    Clock,
    Expr,
    If,
    Input,
    Let,
    Observe,
    Program,
    Region,
    Repeat,
    Require,
    Schedule,
    Store,
    Table,
    Timer,
)

MAX = (1 << 62) - 1
NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,63}\Z")


def integer(x, lo, hi):
    if type(x) is not int or not lo <= x <= hi:
        raise ValueError(f"Expected integer in [{lo}, {hi}]: {x!r}")


def identifier(s):
    if type(s) is not str or NAME.fullmatch(s) is None:
        raise ValueError(f"Invalid name: {s!r}")


def dtype(t):
    if type(t) is not Bits or type(t.signed) is not bool:
        raise ValueError("Expected Bits type")
    integer(t.width, 1, 63)


def rational(s):
    if type(s) is not str:
        raise ValueError("Time must be an exact rational string")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError("Invalid rational time") from e


@dataclass
class Lowered:
    program: Program
    offsets: dict
    width: int
    denominator: int
    periods: tuple
    phases: tuple
    locations: dict
    regions: dict
    observations: dict
    effects: dict
    journal_size: int
    observation_size: int
    records_per_group: int
    digest: str
    table_offsets: dict
    table_width: int
    diagnostics: dict
    timer_initials: tuple
    duration_ticks: tuple
    diagnostic_types: dict


def lower(p):
    if type(p) is not Program:
        raise ValueError("Expected Program")
    identifier(p.name)
    for values in (
        p.cells,
        p.inputs,
        p.clocks,
        p.captures,
        p.schedule,
        p.tables,
        p.timers,
    ):
        if type(values) is not tuple or len(values) > 256:
            raise ValueError(
                "Program collections must be tuples of at most 256 entries"
            )
    integer(p.max_microstep, 1, 1000000)
    cells, inputs, clocks, offsets, timers = {}, {}, {}, {}, {}
    width, denominator = 0, 1
    for values, cls, table in (
        (p.cells, Cell, cells),
        (p.inputs, Input, inputs),
        (p.clocks, Clock, clocks),
    ):
        for v in values:
            if type(v) is not cls:
                raise ValueError(f"Expected {cls.__name__}")
            identifier(v.name)
            if v.name in cells or v.name in inputs or v.name in clocks:
                raise ValueError(f"Duplicate symbol: {v.name}")
            table[v.name] = v
            if cls is Clock:
                period, phase = rational(v.period), rational(v.phase)
                if period <= 0 or phase < 0:
                    raise ValueError(
                        "Clock period must be positive and phase nonnegative"
                    )
                denominator = lcm(denominator, period.denominator, phase.denominator)
            else:
                dtype(v.type)
                if cls is Cell:
                    integer(v.size, 1, 65536)
                    if type(v.initial) is not tuple or len(v.initial) not in (
                        0,
                        v.size,
                    ):
                        raise ValueError("Initial contents must match cell size")
                    for x in v.initial:
                        integer(x, v.type.minimum, v.type.maximum)
                    offsets[v.name] = width
                    width += v.size
                else:
                    integer(v.initial, v.type.minimum, v.type.maximum)
    # Stateless monitors and combinational components need no dummy storage.
    integer(width, 0, 65536)
    tables, table_offsets, table_width = {}, {}, 0
    for t in p.tables:
        if type(t) is not Table:
            raise ValueError("Expected Table")
        identifier(t.name)
        dtype(t.type)
        if t.name in cells or t.name in inputs or t.name in clocks or t.name in tables:
            raise ValueError(f"Duplicate symbol: {t.name}")
        integer(t.size, 1, 16777216)
        if type(t.values) is not tuple or len(t.values) not in (0, t.size):
            raise ValueError("Table values must match declared size")
        for value in t.values:
            integer(value, t.type.minimum, t.type.maximum)
        tables[t.name] = t
        table_offsets[t.name] = table_width
        table_width += t.size
    integer(table_width, 0, 16777216)
    for timer in p.timers:
        if type(timer) is not Timer:
            raise ValueError("Expected Timer")
        identifier(timer.name)
        if timer.name in cells | inputs | clocks | tables | timers:
            raise ValueError(f"Duplicate symbol: {timer.name}")
        integer(timer.microstep, 0, p.max_microstep)
        if timer.initial is not None:
            start = rational(timer.initial)
            if start < 0:
                raise ValueError("Initial timer time must be nonnegative")
            denominator = lcm(denominator, start.denominator)
        timers[timer.name] = timer
    locations, regions, observations, effects, diagnostics = {}, {}, {}, {}, {}
    diagnostic_types = {}
    nodes = 0
    durations = []

    def loc(path, description):
        locations[path] = (len(locations) + 1, description)

    def expression(e, env, effect, depth=0):
        nonlocal nodes, denominator
        nodes += 1
        if nodes > 20000 or depth > 48:
            raise ValueError("Program expression budget exceeded")
        if type(e) is not Expr or type(e.args) is not tuple:
            raise ValueError("Expected a closed Expr")
        dtype(e.type)
        if e.policy not in ("checked", "wrap", "saturate"):
            raise ValueError("Unknown arithmetic conversion policy")
        a, k = e.args, e.op
        if k == "const" and len(a) == 1:
            integer(a[0], e.type.minimum, e.type.maximum)
            return
        if k == "duration" and len(a) == 1 and e.type == TIME:
            value = rational(a[0])
            if value < 0:
                raise ValueError("Duration must be nonnegative")
            denominator = lcm(denominator, value.denominator)
            durations.append(value)
            return
        if k in ("generation", "due_generation") and len(a) == 1:
            if a[0] not in timers or e.type != TIME:
                raise ValueError("Generation requires a declared timer and TIME type")
            effect["event_reads"].add(a[0])
            return
        if k in ("input", "local", "due") and len(a) == 1:
            identifier(a[0])
            table = inputs if k == "input" else env if k == "local" else clocks | timers
            if a[0] not in table:
                raise ValueError(f"Unknown {k}: {a[0]}")
            expected = (
                inputs[a[0]].type
                if k == "input"
                else env[a[0]]
                if k == "local"
                else BOOL
            )
            if e.type != expected:
                raise ValueError(f"Type mismatch for {k}: {a[0]}")
            if k == "due":
                effect["event_reads"].add(a[0])
            return
        if k in ("time", "microstep") and not a and e.type == TIME:
            return
        if k in ("read", "lookup") and len(a) == 2:
            identifier(a[0])
            symbols = cells if k == "read" else tables
            if a[0] not in symbols or symbols[a[0]].type != e.type:
                raise ValueError(f"Unknown cell or read type mismatch: {a[0]}")
            expression(a[1], env, effect, depth + 1)
            effect["reads"].add(a[0])
            return
        arity = {
            "cast": 1,
            "not": 1,
            "add": 2,
            "sub": 2,
            "mul": 2,
            "eq": 2,
            "ne": 2,
            "lt": 2,
            "le": 2,
            "gt": 2,
            "ge": 2,
            "and": 2,
            "or": 2,
            "select": 3,
            "round_even": 2,
            "round_away": 2,
            "round_floor": 2,
            "round_zero": 2,
            "div": 2,
            "mod": 2,
        }
        if k not in arity or len(a) != arity[k]:
            raise ValueError(f"Unsupported expression: {k}")
        for child in a:
            expression(child, env, effect, depth + 1)
        if k in ("and", "or", "not"):
            if e.type != BOOL or any(x.type != BOOL for x in a):
                raise ValueError("Boolean operation requires BOOL")
        elif k in ("eq", "ne", "lt", "le", "gt", "ge"):
            if a[0].type != a[1].type or e.type != BOOL:
                raise ValueError(
                    "Comparison requires matching operand types and BOOL result"
                )
        elif k == "select":
            if a[0].type != BOOL or a[1].type != e.type or a[2].type != e.type:
                raise ValueError("Select requires BOOL and matching branch types")
        elif k in ("round_even", "round_away", "round_floor", "round_zero"):
            if a[1].op != "const":
                raise ValueError("Rounding shift must be a constant")
            integer(a[1].args[0], 0, 61)
        elif k in ("add", "sub", "mul"):
            # Literal ranges are exact; other expressions retain their declared
            # type range. This admits multiplication by a small constant safely.
            def bounds(x):
                return (
                    (x.args[0], x.args[0])
                    if x.op == "const"
                    else (x.type.minimum, x.type.maximum)
                )

            x, y = bounds(a[0]), bounds(a[1])
            if k == "add":
                bounds = (x[0] + y[0], x[1] + y[1])
            elif k == "sub":
                bounds = (x[0] - y[1], x[1] - y[0])
            else:
                bounds = tuple(i * j for i in x for j in y)
            if min(bounds) < -(1 << 63) + 1 or max(bounds) > (1 << 63) - 1:
                raise ValueError(
                    "Arithmetic intermediate exceeds safe native carrier; narrow operands explicitly"
                )

    def boolean(e, env, effect):
        expression(e, env, effect)
        if e.type != BOOL:
            raise ValueError("Guard/condition must be BOOL")

    def statements(body, env, effect, path, depth=0):
        if depth > 32 or type(body) is not tuple or len(body) > 1024:
            raise ValueError("Invalid or excessive statement block")
        writes, obs = 0, 0
        for i, s in enumerate(body):
            key = f"{path}/{i}"
            loc(key, s.message if type(s) is Require else type(s).__name__)
            if type(s) is Let:
                identifier(s.name)
                if s.name in env:
                    raise ValueError(f"Local already defined: {s.name}")
                expression(s.value, env, effect)
                env[s.name] = s.value.type
            elif type(s) is Store:
                if s.target not in cells or s.value.type != cells[s.target].type:
                    raise ValueError(f"Store type mismatch: {s.target}")
                expression(s.index, env, effect)
                expression(s.value, env, effect)
                effect["writes"].add(s.target)
                writes += 1
            elif type(s) in (Schedule, Cancel):
                if s.timer not in timers:
                    raise ValueError(f"Unknown timer: {s.timer}")
                if type(s) is Schedule:
                    for value in (s.delay, s.microstep):
                        expression(value, env, effect)
                        if value.type != TIME:
                            raise ValueError("Timer delay and microstep require TIME")
                effect["events"].add(s.timer)
            elif type(s) is Require:
                if type(s.message) is not str or len(s.message) > 256:
                    raise ValueError("Requirement message must be a short string")
                boolean(s.condition, env, effect)
                if type(s.details) is not tuple or len(s.details) > 8:
                    raise ValueError(
                        "Requirement supports at most eight diagnostic values"
                    )
                for value in s.details:
                    expression(value, env, effect)
                diagnostics[key] = len(s.details)
                diagnostic_types[key] = tuple(value.type for value in s.details)
            elif type(s) is Observe:
                identifier(s.name)
                if type(s.values) is not tuple or len(s.values) > 8:
                    raise ValueError("Observation supports at most eight scalar values")
                signature = tuple(x.type for x in s.values)
                if s.name in observations and observations[s.name][1] != signature:
                    raise ValueError("Observation signatures must agree")
                observations.setdefault(s.name, (len(observations) + 1, signature))
                for x in s.values:
                    expression(x, env, effect)
                effect["observations"].add(s.name)
                obs += 1
            elif type(s) is If:
                boolean(s.condition, env, effect)
                counts = [
                    statements(b, env.copy(), effect, key + suffix, depth + 1)
                    for b, suffix in ((s.then, "/then"), (s.otherwise, "/else"))
                ]
                writes += max(c[0] for c in counts)
                obs += max(c[1] for c in counts)
            else:
                raise ValueError(f"Unsupported statement: {type(s).__name__}")
        return writes, obs

    def effect():
        return {
            "reads": set(),
            "writes": set(),
            "observations": set(),
            "event_reads": set(),
            "events": set(),
            "may_fault": True,
        }

    env = {}
    cap_effect = effect()
    if any(type(s) is not Let for s in p.captures):
        raise ValueError("Group captures must be scalar Let bindings")
    statements(p.captures, env, cap_effect, "capture")
    effects["capture"] = cap_effect
    journal, staged, records = 1, 1, 1

    def region(r, path):
        nonlocal journal, staged
        if type(r) is not Region:
            raise ValueError("Repeat body must contain Regions")
        identifier(r.name)
        if r.name in regions:
            raise ValueError(f"Duplicate region: {r.name}")
        regions[r.name] = (len(regions) + 1, path)
        e = effect()
        loc(path, r.name)
        boolean(r.guard, env, e)
        w, o = statements(r.body, env.copy(), e, path)
        journal, staged = max(journal, w), max(staged, o)
        effects[r.name] = e
        return o + 1

    names = set()
    for i, item in enumerate(p.schedule):
        path = f"schedule/{i}"
        if type(item) is Repeat:
            identifier(item.name)
            if item.name in names:
                raise ValueError("Duplicate schedule name")
            integer(item.limit, 1, 1024)
            if type(item.regions) is not tuple or not item.regions:
                raise ValueError("Repeat requires a nonempty tuple of regions")
            loc(path, item.name)
            e = effect()
            boolean(item.condition, env, e)
            effects[item.name] = e
            records += (
                item.limit
                * sum(
                    region(r, f"{path}/region/{j}") for j, r in enumerate(item.regions)
                )
                + 1
            )
        elif type(item) is Region:
            records += region(item, path)
        else:
            raise ValueError("Schedule requires Regions or bounded Repeats")
        names.add(item.name)
    if journal > 4096 or staged > 1024 or records > 100000:
        raise ValueError("Program transaction/log budget exceeded")
    loc("stop", "completion condition")
    effects["stop"] = effect()
    boolean(p.stop, {}, effects["stop"])
    integer(denominator, 1, MAX)
    periods = tuple(int(rational(c.period) * denominator) for c in p.clocks)
    phases = tuple(int(rational(c.phase) * denominator) for c in p.clocks)
    timer_initials = tuple(
        None if t.initial is None else int(rational(t.initial) * denominator)
        for t in p.timers
    )
    for value in periods + phases + tuple(t for t in timer_initials if t is not None):
        integer(value, 0, MAX)
    duration_ticks = tuple(int(t * denominator) for t in durations)
    for value in duration_ticks:
        integer(value, 0, MAX)
    digest = hashlib.sha256(
        json.dumps(
            {"schema": SCHEMA, "program": asdict(p)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return Lowered(
        p,
        offsets,
        width,
        denominator,
        periods,
        phases,
        locations,
        regions,
        observations,
        effects,
        journal,
        staged,
        records,
        digest,
        table_offsets,
        table_width,
        diagnostics,
        timer_initials,
        duration_ticks,
        diagnostic_types,
    )
