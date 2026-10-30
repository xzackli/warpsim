"""Direct semantic interpreter using exact rational events and copied snapshots.

It walks the semantic tree independently of the native emitter, using Python
integers and dictionaries. Only validation, input validation and public records
are shared with native execution.
"""

from fractions import Fraction

from .execution import (
    FAULTS,
    Record,
    Result,
    check_memory,
    prepare,
    state_dict,
    trace_layout,
)
from .ir import Cancel, If, Let, Observe, Region, Require, Schedule, Store
from .lower import MAX, lower, rational


class Fault(Exception):
    def __init__(self, code, location, target=None, index=None, values=()):
        self.code, self.location, self.target, self.index = (
            code,
            location,
            target,
            index,
        )
        self.values = values


class Interpreter:
    def __init__(self, l, prepared, trace_cells=None, trace_mode="full"):
        self.l, self.p, self.prepared = l, l.program, prepared
        self.state = state_dict(l, prepared.initial)
        self.inputs = {x.name: x.initial for x in self.p.inputs}
        self.records = []
        self.time = None
        self.due = set()
        self.microstep = 0
        self.iteration = None
        self.trace_mode = trace_mode
        self.counts = [0] * len(self.p.clocks)
        self.cursor, self.events, self.status = 0, 0, None
        self.timers = {
            t.name: None if t.initial is None else (rational(t.initial), t.microstep)
            for t in self.p.timers
        }
        self.generations = {t.name: 0 for t in self.p.timers}
        self.due_generations = {}
        self.trace_names, _ = trace_layout(l, trace_cells)
        self.tables = {
            t.name: prepared.tables[
                l.table_offsets[t.name] : l.table_offsets[t.name] + t.size
            ]
            for t in self.p.tables
        }

    def expr(self, e, snapshot, env, path):
        k, a = e.op, e.args
        if k == "const":
            return a[0]
        if k in ("read", "lookup"):
            i = self.expr(a[1], snapshot, env, path)
            source = snapshot if k == "read" else self.tables
            if not 0 <= i < len(source[a[0]]):
                raise Fault(1, path, a[0], i)
            return source[a[0]][i]
        if k == "input":
            return self.inputs[a[0]]
        if k == "local":
            return env[a[0]]
        if k == "due":
            return int(a[0] in self.due)
        if k == "time":
            return int(self.time * self.prepared.denominator)
        if k == "microstep":
            return self.microstep
        if k == "duration":
            value = int(rational(a[0]) * self.prepared.denominator)
            if value > MAX:
                raise Fault(2, path)
            return value
        if k in ("generation", "due_generation"):
            return (self.generations if k == "generation" else self.due_generations)[
                a[0]
            ]
        if k == "select":
            return self.expr(
                a[1] if self.expr(a[0], snapshot, env, path) else a[2],
                snapshot,
                env,
                path,
            )
        # and/or are eager; unsafe operations belong under If or select.
        v = [self.expr(x, snapshot, env, path) for x in a]
        if k == "cast":
            value = v[0]
        elif k in ("round_even", "round_away", "round_floor", "round_zero"):
            divisor = 1 << v[1]
            q, r = divmod(v[0], divisor)
            if k == "round_even":
                value = q + int(2 * r > divisor or (2 * r == divisor and q % 2 != 0))
            elif k == "round_floor":
                value = q
            else:
                magnitude, remainder = divmod(abs(v[0]), divisor)
                magnitude += int(k == "round_away" and 2 * remainder >= divisor)
                value = -magnitude if v[0] < 0 else magnitude
        elif k in ("div", "mod"):
            if v[1] == 0:
                raise Fault(6, path)
            q = abs(v[0]) // abs(v[1])
            if (v[0] < 0) != (v[1] < 0):
                q = -q
            value = q if k == "div" else v[0] - q * v[1]
        elif k in ("add", "sub", "mul"):
            value = {
                "add": lambda: v[0] + v[1],
                "sub": lambda: v[0] - v[1],
                "mul": lambda: v[0] * v[1],
            }[k]()
        else:
            return int(
                {
                    "eq": lambda: v[0] == v[1],
                    "ne": lambda: v[0] != v[1],
                    "lt": lambda: v[0] < v[1],
                    "le": lambda: v[0] <= v[1],
                    "gt": lambda: v[0] > v[1],
                    "ge": lambda: v[0] >= v[1],
                    "and": lambda: bool(v[0] and v[1]),
                    "or": lambda: bool(v[0] or v[1]),
                    "not": lambda: not v[0],
                }[k]()
            )
        t = e.type
        if e.policy == "checked":
            if not t.minimum <= value <= t.maximum:
                raise Fault(2, path)
            return value
        if e.policy == "saturate":
            return min(t.maximum, max(t.minimum, value))
        value %= 1 << t.width
        return value - (1 << t.width) if t.signed and value > t.maximum else value

    def body(self, statements, snapshot, env, path, patch, observations, event_patch):
        for i, s in enumerate(statements):
            key = f"{path}/{i}"
            if type(s) is Let:
                env[s.name] = self.expr(s.value, snapshot, env, key)
            elif type(s) is Store:
                index = self.expr(s.index, snapshot, env, key)
                if not 0 <= index < len(snapshot[s.target]):
                    raise Fault(1, key, s.target, index)
                value = self.expr(s.value, snapshot, env, key)
                if (s.target, index) in patch:
                    raise Fault(3, key, s.target, index)
                patch[s.target, index] = value
            elif type(s) is Require:
                if not self.expr(s.condition, snapshot, env, key):
                    details = tuple(self.expr(x, snapshot, env, key) for x in s.details)
                    raise Fault(4, key, values=details)
            elif type(s) in (Schedule, Cancel):
                tag = None
                if type(s) is Schedule:
                    delay = self.expr(s.delay, snapshot, env, key)
                    step = self.expr(s.microstep, snapshot, env, key)
                    tick = int(self.time * self.prepared.denominator) + delay
                    if tick > MAX:
                        raise Fault(2, key, s.timer, 0)
                    if delay == 0 and step <= self.microstep:
                        raise Fault(7, key, s.timer, 0)
                    if step > self.p.max_microstep:
                        raise Fault(8, key, s.timer, 0)
                    tag = (Fraction(tick, self.prepared.denominator), step)
                if s.timer in event_patch:
                    raise Fault(3, key, s.timer, 0)
                if self.generations[s.timer] == MAX:
                    raise Fault(2, key, s.timer, 0)
                event_patch[s.timer] = tag
            elif type(s) is Observe:
                observations.append(
                    (
                        key,
                        s.name,
                        tuple(self.expr(x, snapshot, env, key) for x in s.values),
                    )
                )
            elif type(s) is If:
                enabled = self.expr(s.condition, snapshot, env, key)
                self.body(
                    s.then if enabled else s.otherwise,
                    snapshot,
                    env.copy(),
                    key + ("/then" if enabled else "/else"),
                    patch,
                    observations,
                    event_patch,
                )

    def record(
        self, kind, region=None, location=None, observation=None, values=(), fault=None
    ):
        if self.trace_mode == "summary" and kind != "fault":
            return
        if self.trace_mode == "observations" and kind not in ("observe", "fault"):
            return
        if fault:
            values = fault.values
        self.records.append(
            Record(
                self.time,
                kind,
                region,
                location,
                observation,
                values,
                FAULTS[fault.code] if fault else None,
                fault.target if fault else None,
                fault.index if fault else None,
                self.l.locations[location][1] if fault else None,
                {name: self.state[name] for name in self.trace_names},
                self.microstep,
                self.iteration,
            )
        )

    def region(self, r, env, path):
        self.active_region = r.name
        snapshot = self.state.copy()
        if not self.expr(r.guard, snapshot, env, path):
            self.record("disabled", r.name, path)
            return
        patch, observations, event_patch = {}, [], {}
        self.body(r.body, snapshot, env.copy(), path, patch, observations, event_patch)
        updated = {name: list(values) for name, values in self.state.items()}
        for (name, index), value in patch.items():
            updated[name][index] = value
        self.state = {name: tuple(values) for name, values in updated.items()}
        for name, tag in event_patch.items():
            self.timers[name] = tag
            self.generations[name] += 1
        self.record("commit", r.name, path)
        for key, name, values in observations:
            self.record("observe", r.name, key, name, values)

    def group(self):
        env = {}
        self.active_region = None
        try:
            self.iteration = None
            self.body(self.p.captures, self.state, env, "capture", {}, [], {})
            for i, item in enumerate(self.p.schedule):
                self.iteration = None
                path = f"schedule/{i}"
                if type(item) is Region:
                    self.region(item, env, path)
                else:
                    for iteration in range(item.limit + 1):
                        self.iteration = iteration
                        self.active_region = None
                        if not self.expr(item.condition, self.state, env, path):
                            break
                        if iteration == item.limit:
                            raise Fault(5, path)
                        for j, r in enumerate(item.regions):
                            self.region(r, env, f"{path}/region/{j}")
            self.active_region = None
            self.iteration = None
            if self.expr(self.p.stop, self.state, {}, "stop"):
                return "complete"
        except Fault as fault:
            self.record("fault", self.active_region, fault.location, fault=fault)
            return False
        return True

    def run(self, until=None, max_events=None):
        self.records = []
        horizon = Fraction(
            self.prepared.until if until is None else until, self.prepared.denominator
        )
        budget = self.prepared.max_events if max_events is None else max_events
        processed = 0
        drives = self.prepared.drives
        while self.status not in ("fault", "complete"):
            clocks = [
                (rational(c.phase) + k * rational(c.period), 0)
                for c, k in zip(self.p.clocks, self.counts)
            ]
            tags = clocks + [tag for tag in self.timers.values() if tag is not None]
            if self.cursor < len(drives):
                tags.append(
                    (Fraction(drives[self.cursor][0], self.prepared.denominator), 0)
                )
            if not tags or min(tags)[0] > horizon:
                self.status = "horizon"
                break
            if processed == budget:
                self.status = "budget"
                break
            self.time, self.microstep = tag = min(tags)
            self.due = {c.name for c, t in zip(self.p.clocks, clocks) if t == tag}
            self.due_generations = self.generations.copy()
            for name, pending in self.timers.items():
                if pending == tag:
                    self.due.add(name)
                    self.timers[name] = None
            self.counts = [
                k + int(c.name in self.due) for c, k in zip(self.p.clocks, self.counts)
            ]
            if (
                self.microstep == 0
                and self.cursor < len(drives)
                and Fraction(drives[self.cursor][0], self.prepared.denominator)
                == self.time
            ):
                self.inputs = {
                    x.name: v for x, v in zip(self.p.inputs, drives[self.cursor][1:])
                }
                self.cursor += 1
            self.events += 1
            processed += 1
            outcome = self.group()
            if not outcome:
                self.status = "fault"
                break
            if outcome == "complete":
                self.status = "complete"
                break
        calendar = {
            "clocks": {
                c.name: int(
                    (rational(c.phase) + k * rational(c.period))
                    * self.prepared.denominator
                )
                for c, k in zip(self.p.clocks, self.counts)
            },
            "timers": {
                name: (
                    None
                    if tag is None
                    else (int(tag[0] * self.prepared.denominator), tag[1]),
                    self.generations[name],
                )
                for name, tag in self.timers.items()
            },
            "inputs": self.inputs.copy(),
            "cursor": self.cursor,
        }
        calendar["clocks"] = {
            name: t if t <= MAX else None for name, t in calendar["clocks"].items()
        }
        return Result(
            self.l.digest,
            Fraction(1, self.prepared.denominator),
            self.status,
            self.time,
            self.events,
            self.state.copy(),
            tuple(self.records),
            self.microstep,
            calendar,
        )


def run_reference(
    program,
    *,
    until,
    drives=(),
    initial=None,
    max_events=256,
    tables=None,
    trace_cells=None,
    memory_limit=256 * 1024**2,
    trace_mode="full",
):
    l = lower(program)
    prepared = prepare(l, until, drives, initial, max_events, tables)
    check_memory(l, prepared, trace_cells, memory_limit, trace_mode)
    return Interpreter(l, prepared, trace_cells, trace_mode).run()
