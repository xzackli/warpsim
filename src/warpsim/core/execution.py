"""Shared public records and validated run inputs; no transition evaluation."""

from dataclasses import dataclass, field
from fractions import Fraction

from .ir import Drive
from .lower import MAX, integer, rational

FAULTS = {
    1: "bounds",
    2: "overflow",
    3: "conflict",
    4: "requirement",
    5: "loop_limit",
    6: "division_by_zero",
    7: "event_order",
    8: "microstep_limit",
}


@dataclass(frozen=True)
class Record:
    time: Fraction
    kind: str
    region: str | None
    location: str | None
    observation: str | None
    values: tuple
    fault: str | None
    target: str | None
    index: int | None
    message: str | None
    state: dict
    microstep: int = 0
    iteration: int | None = None


@dataclass(frozen=True)
class Result:
    model_hash: str
    tick_seconds: Fraction
    status: str
    time: Fraction | None
    events: int
    state: dict
    records: tuple
    microstep: int = 0
    calendar: dict = field(default_factory=dict)

    @property
    def fault(self):
        return next((r for r in reversed(self.records) if r.kind == "fault"), None)


@dataclass
class Prepared:
    until: int
    max_events: int
    initial: list
    drives: list
    denominator: int
    tables: list


def prepare(l, until, drives, initial, max_events, tables=None, *, time_denominator=1):
    """Extend the exact clock lattice for the supplied drive timestamps."""
    from math import lcm

    integer(max_events, 1, 100000)
    horizon = rational(until)
    if horizon < 0:
        raise ValueError("Horizon must be nonnegative")
    integer(time_denominator, 1, MAX)
    denominator = lcm(l.denominator, time_denominator)
    parsed = []
    previous = Fraction(-1)
    if type(drives) not in (tuple, list) or len(drives) > 100000:
        raise ValueError("Drives must be a bounded ordered sequence")
    symbols = {x.name: x for x in l.program.inputs}
    for d in drives:
        if type(d) is not Drive or type(d.values) is not dict:
            raise ValueError("Expected Drive records")
        t = rational(d.time)
        if t < 0 or t <= previous:
            raise ValueError(
                "Drive times must be nonnegative and strictly increasing; merge simultaneous changes"
            )
        previous = t
        denominator = lcm(denominator, t.denominator)
        for name, v in d.values.items():
            if name not in symbols:
                raise ValueError(f"Unknown input: {name}")
            typ = symbols[name].type
            integer(v, typ.minimum, typ.maximum)
        parsed.append((t, d.values))
    integer(denominator, 1, MAX)
    # No event can occur between the final lattice tick and the inclusive horizon.
    ticks = horizon * denominator
    integer(ticks.numerator // ticks.denominator, 0, MAX)
    scale = denominator // l.denominator
    for x in (
        l.phases
        + l.periods
        + l.duration_ticks
        + tuple(x for x in l.timer_initials if x is not None)
    ):
        integer(x * scale, 0, MAX)
    held = [i.initial for i in l.program.inputs]
    rows = []
    for t, changes in parsed:
        tick = int(t * denominator)
        integer(tick, 0, MAX)
        held = [changes.get(i.name, v) for i, v in zip(l.program.inputs, held)]
        rows.append([tick, *held])
    if initial is None:
        initial = {}
    if type(initial) is not dict or set(initial) - set(l.offsets):
        raise ValueError("Unknown initial state override")
    values = []
    for cell in l.program.cells:
        data = initial.get(cell.name, cell.initial or (0,) * cell.size)
        if type(data) is int and cell.size == 1:
            data = (data,)
        if type(data) not in (tuple, list) or len(data) != cell.size:
            raise ValueError(f"Initial override must match {cell.name} size")
        for x in data:
            integer(x, cell.type.minimum, cell.type.maximum)
        values.extend(data)
    tables = {} if tables is None else tables
    if type(tables) is not dict or set(tables) - set(l.table_offsets):
        raise ValueError("Unknown table override")
    table_values = []
    for t in l.program.tables:
        if t.values and t.name in tables:
            raise ValueError(f"Constant table cannot be overridden: {t.name}")
        data = t.values or tables.get(t.name, ())
        if len(data) != t.size:
            raise ValueError(f"Missing or incorrectly sized table: {t.name}")
        for value in data:
            integer(value, t.type.minimum, t.type.maximum)
        table_values.extend(data)
    return Prepared(int(ticks), max_events, values, rows, denominator, table_values)


def state_dict(l, values):
    return {
        c.name: tuple(
            int(v) for v in values[l.offsets[c.name] : l.offsets[c.name] + c.size]
        )
        for c in l.program.cells
    }


def trace_layout(l, trace_cells):
    names = tuple(l.offsets) if trace_cells is None else trace_cells
    if (
        type(names) is not tuple
        or len(set(names)) != len(names)
        or set(names) - set(l.offsets)
    ):
        raise ValueError("trace_cells must be a tuple of distinct state cell names")
    cells = {c.name: c for c in l.program.cells}
    indices = tuple(
        i
        for name in names
        for i in range(l.offsets[name], l.offsets[name] + cells[name].size)
    )
    return names, indices


def check_memory(l, prepared, trace_cells, memory_limit, trace_mode="full"):
    integer(memory_limit, 1024, 64 * 1024**3)
    if trace_mode not in ("full", "observations", "summary"):
        raise ValueError("trace_mode must be full, observations, or summary")
    _, indices = trace_layout(l, trace_cells)
    count = 1 if trace_mode == "summary" else prepared.max_events * l.records_per_group
    words = count * (18 + len(indices))
    words += (
        l.width
        + l.table_width
        + (len(l.program.inputs) + 1) * max(1, len(prepared.drives))
        + 12
        + 3 * len(l.program.timers)
        + len(l.program.clocks)
        + len(l.program.inputs)
        + 2  # Per-scenario external count and time-lattice scale.
    )
    if words * 8 > memory_limit:
        raise ValueError(
            "Core packed buffers exceed memory budget; reduce events or trace cells"
        )


def decode_record(l, denominator, row, trace_cells=None):
    time, kind, region, location, code, target, index, obs = map(int, row[:8])
    region_name = next((k for k, v in l.regions.items() if v[0] == region), None)
    location_name = next((k for k, v in l.locations.items() if v[0] == location), None)
    observation = next((k for k, v in l.observations.items() if v[0] == obs), None)
    target_name = (
        next((k for k, v in l.offsets.items() if v == target), None)
        if target >= 0
        else next((k for k, v in l.table_offsets.items() if -2 - v == target), None)
    )
    if target <= -2 - l.table_width:
        timer = -2 - l.table_width - target
        if timer < len(l.program.timers):
            target_name = l.program.timers[timer].name
    size = len(l.observations[observation][1]) if observation else 0
    if kind == 4:
        size = l.diagnostics.get(location_name, 0) if code == 4 else 0
    names, _ = trace_layout(l, trace_cells)
    cells = {c.name: c for c in l.program.cells}
    state, offset = {}, 18
    for name in names:
        state[name] = tuple(map(int, row[offset : offset + cells[name].size]))
        offset += cells[name].size
    return Record(
        Fraction(time, denominator),
        {1: "commit", 2: "disabled", 3: "observe", 4: "fault"}[kind],
        region_name,
        location_name,
        observation,
        tuple(map(int, row[8 : 8 + size])),
        FAULTS.get(code),
        target_name,
        index if target != -1 else None,
        l.locations[location_name][1] if code and location_name else None,
        state,
        int(row[16]),
        int(row[17]) if row[17] >= 0 else None,
    )


def data_offset(program):
    """Executor frontier, periodic sources, timer slots, inputs, then model state."""
    return 12 + len(program.clocks) + 3 * len(program.timers) + len(program.inputs)
