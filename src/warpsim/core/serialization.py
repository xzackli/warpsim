"""Versioned closed-program cases and portable event-boundary checkpoints.

No pickle, imports named by input files, executable callbacks or model registry.
Only explicitly listed data constructors can cross this boundary.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction
from pathlib import Path

from . import ir
from .execution import FAULTS, Record, Result, prepare, trace_layout
from .lower import MAX, integer, lower, rational
from .session import Scenario, Session


@dataclass(frozen=True)
class Case:
    program: ir.Program
    until: str
    scenarios: tuple = (Scenario(),)
    max_events: int = 256
    trace_cells: tuple | None = None
    time_denominator: int = 1

    def validate(self):
        l = lower(self.program)
        integer(self.max_events, 1, MAX)
        if type(self.scenarios) is not tuple or not 1 <= len(self.scenarios) <= 65536:
            raise ValueError("Case requires a bounded tuple of scenarios")
        for s in self.scenarios:
            if type(s) is not Scenario:
                raise ValueError("Expected Scenario")
            prepare(
                l,
                self.until,
                s.drives,
                s.initial,
                min(100000, self.max_events),
                s.tables,
                time_denominator=self.time_denominator,
            )
        from .execution import trace_layout

        trace_layout(l, self.trace_cells)
        return self

    def run(self, *, device="cpu", **options):
        self.validate()
        session = Session(
            self.program,
            self.scenarios,
            device=device,
            trace_cells=self.trace_cells,
            time_denominator=self.time_denominator,
            **options,
        )
        return session.run(until=self.until, max_events=self.max_events)

    def save(self, path):
        self.validate()
        save(path, "warpsim.case/v2", self)

    @classmethod
    def load(cls, path):
        case = load(path, "warpsim.case/v2")
        if type(case) is not cls:
            raise ValueError("Expected a semantic Case")
        return case.validate()


_TYPES = {
    cls.__name__: cls
    for cls in (
        ir.Bits,
        ir.Expr,
        ir.Cell,
        ir.Input,
        ir.Table,
        ir.Clock,
        ir.Timer,
        ir.Schedule,
        ir.Cancel,
        ir.Let,
        ir.Store,
        ir.Require,
        ir.Observe,
        ir.If,
        ir.Region,
        ir.Repeat,
        ir.Program,
        ir.Drive,
        Scenario,
        Case,
        Result,
        Record,
    )
}


def encode(value):
    if is_dataclass(value) and type(value).__name__ in _TYPES:
        return {
            "$type": type(value).__name__,
            **{f.name: encode(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, Fraction):
        return {"$fraction": str(value)}
    if type(value) is tuple:
        return {"$tuple": [encode(x) for x in value]}
    if type(value) is list:
        return [encode(x) for x in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: encode(v) for k, v in value.items()}
    if value is None or type(value) in (str, int, bool):
        return value
    raise ValueError(f"Unsupported serialized value: {type(value).__name__}")


def decode(value, depth=0):
    if depth > 128:
        raise ValueError("Serialized nesting limit exceeded")
    child = lambda v: decode(v, depth + 1)
    if type(value) is list:
        return [child(x) for x in value]
    if type(value) is dict:
        if "$type" in value:
            cls = _TYPES.get(value["$type"])
            if cls is None or set(value) != {"$type", *(f.name for f in fields(cls))}:
                raise ValueError("Unknown data constructor or fields")
            return cls(**{k: child(v) for k, v in value.items() if k != "$type"})
        if set(value) == {"$tuple"} and type(value["$tuple"]) is list:
            return tuple(child(x) for x in value["$tuple"])
        if set(value) == {"$fraction"}:
            return rational(value["$fraction"])
        return {k: child(v) for k, v in value.items()}
    if value is None or type(value) in (str, int, bool):
        return value
    raise ValueError("Non-data JSON value")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path, schema, value):
    data = encode(value)
    document = {
        "schema": schema,
        "sha256": hashlib.sha256(canonical(data).encode()).hexdigest(),
        "data": data,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            name = f.name
            json.dump(document, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def load(path, schema):
    path = Path(path)
    if path.stat().st_size > 512 * 1024**2:
        raise ValueError("Serialized artifact exceeds size limit")
    document = json.loads(path.read_text())
    if (
        type(document) is not dict
        or set(document) != {"schema", "sha256", "data"}
        or document["schema"] != schema
        or hashlib.sha256(canonical(document["data"]).encode()).hexdigest()
        != document["sha256"]
    ):
        raise ValueError("Artifact schema or checksum mismatch")
    return decode(document["data"])


def save_checkpoint(session, path):
    if session.results is None:
        raise ValueError("Step the session before checkpointing")
    save(
        path,
        "warpsim.checkpoint/v2",
        {
            "program": session.program,
            "scenarios": session.scenarios,
            "until": str(session.until),
            "denominator": session.denominator,
            "trace_cells": session.trace_cells,
            "trace_mode": session.trace_mode,
            "results": tuple(
                replace(r, records=(r.fault,) if r.fault else ())
                for r in session.results
            ),
        },
    )


def _validate_result(session, result, prepared):
    p, l = session.program, session.l
    if (
        type(result) is not Result
        or result.model_hash != l.digest
        or result.tick_seconds != Fraction(1, session.denominator)
        or result.status not in ("horizon", "budget", "fault", "complete")
    ):
        raise ValueError("Checkpoint identity/status mismatch")
    integer(result.events, 0, MAX)
    integer(result.microstep, 0, p.max_microstep)
    if (result.events == 0) != (result.time is None):
        raise ValueError("Checkpoint event count and frontier disagree")
    if result.time is not None:
        if type(result.time) is not Fraction:
            raise ValueError("Checkpoint time must be exact")
        ticks = result.time * session.denominator
        if ticks.denominator != 1:
            raise ValueError("Checkpoint time is off lattice")
        integer(int(ticks), 0, MAX)
    if set(result.state) != set(l.offsets):
        raise ValueError("Checkpoint state shape mismatch")
    if type(result.records) is not tuple:
        raise ValueError("Checkpoint records must be a tuple")
    if result.status == "fault":
        if len(result.records) != 1 or type(result.records[0]) is not Record:
            raise ValueError("Checkpoint fault requires one diagnostic record")
        fault = result.records[0]
        names, _ = trace_layout(l, session.trace_cells)
        if (
            fault.kind != "fault"
            or fault.time != result.time
            or fault.microstep != result.microstep
            or fault.observation is not None
            or fault.fault not in FAULTS.values()
            or fault.location not in l.locations
            or fault.message != l.locations[fault.location][1]
            or fault.region is not None
            and fault.region not in l.regions
            or fault.state != {name: result.state[name] for name in names}
        ):
            raise ValueError("Checkpoint diagnostic disagrees with its frontier/state")
        size = (
            l.diagnostics.get(fault.location, 0) if fault.fault == "requirement" else 0
        )
        if type(fault.values) is not tuple or len(fault.values) != size:
            raise ValueError("Checkpoint diagnostic operands disagree with the program")
        for value, typ in zip(fault.values, l.diagnostic_types.get(fault.location, ())):
            integer(value, typ.minimum, typ.maximum)
    elif result.records:
        raise ValueError("Checkpoint retains only a terminal fault record")
    runtime_tables = {
        t.name: prepared.tables[
            l.table_offsets[t.name] : l.table_offsets[t.name] + t.size
        ]
        for t in p.tables
        if not t.values
    }
    values = prepare(
        l,
        "0",
        (),
        result.state,
        1,
        runtime_tables,
        time_denominator=session.denominator,
    ).initial
    cal = result.calendar
    if (
        set(cal) != {"clocks", "timers", "inputs", "cursor"}
        or set(cal["clocks"]) != {c.name for c in p.clocks}
        or set(cal["timers"]) != {t.name for t in p.timers}
        or set(cal["inputs"]) != {i.name for i in p.inputs}
    ):
        raise ValueError("Checkpoint calendar shape mismatch")
    integer(cal["cursor"], 0, len(prepared.drives))
    expected_cursor = sum(
        result.time is not None and Fraction(d[0], session.denominator) <= result.time
        for d in prepared.drives
    )
    if cal["cursor"] != expected_cursor:
        raise ValueError("Checkpoint input cursor disagrees with its frontier")
    expected_inputs = (
        prepared.drives[cal["cursor"] - 1][1:]
        if cal["cursor"]
        else [i.initial for i in p.inputs]
    )
    if list(cal["inputs"].get(i.name) for i in p.inputs) != expected_inputs:
        raise ValueError("Checkpoint held inputs disagree with consumed drives")
    frontier = (
        None
        if result.time is None
        else (int(result.time * session.denominator), result.microstep)
    )

    def future(t, step):
        integer(t, 0, MAX)
        integer(step, 0, p.max_microstep)
        if frontier is not None and (t, step) <= frontier:
            raise ValueError("Checkpoint pending event is behind its frontier")

    for c in p.clocks:
        tick = cal["clocks"][c.name]
        occurrence = (
            0
            if result.time is None
            else max(0, (result.time - rational(c.phase)) // rational(c.period) + 1)
        )
        expected = int(
            (rational(c.phase) + occurrence * rational(c.period)) * session.denominator
        )
        expected = expected if expected <= MAX else None
        if tick != expected:
            raise ValueError("Checkpoint clock disagrees with its frontier")
        if tick is not None:
            future(tick, 0)
            occurrence = (
                Fraction(tick, session.denominator) - rational(c.phase)
            ) / rational(c.period)
            if occurrence < 0 or occurrence.denominator != 1:
                raise ValueError("Checkpoint clock is off its occurrence sequence")
    for tag, gen in cal["timers"].values():
        integer(gen, 0, MAX)
        if tag is not None:
            future(*tag)
    if cal["cursor"] < len(prepared.drives):
        future(prepared.drives[cal["cursor"]][0], 0)
    return values


def restore_checkpoint(
    path,
    *,
    device="cpu",
    mode="release",
    memory_limit=256 * 1024**2,
    **execution_options,
):
    if (
        device == "reference"
        and execution_options.get("strategy", "scalar") != "scalar"
    ):
        raise ValueError("Cooperative strategies require CUDA")
    data = load(path, "warpsim.checkpoint/v2")
    if type(data) is not dict or set(data) != {
        "program",
        "scenarios",
        "until",
        "denominator",
        "trace_cells",
        "trace_mode",
        "results",
    }:
        raise ValueError("Invalid checkpoint fields")
    # Validate portable data before loading native code or allocating device state.
    session = Session(
        data["program"],
        data["scenarios"],
        device="reference",
        mode=mode,
        memory_limit=memory_limit,
        trace_cells=data["trace_cells"],
        trace_mode=data["trace_mode"],
        time_denominator=data["denominator"],
        **{k: v for k, v in execution_options.items() if k != "strategy"},
    )
    if data["denominator"] != session.denominator or len(data["results"]) != len(
        session.scenarios
    ):
        raise ValueError("Checkpoint batch/lattice mismatch")
    try:
        values = [
            _validate_result(session, r, p)
            for r, p in zip(data["results"], session.prepared)
        ]
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed checkpoint state or calendar") from error
    until = rational(data["until"])
    if (
        until < 0
        or int(until * session.denominator) > MAX
        or any(r.time is not None and r.time > until for r in data["results"])
    ):
        raise ValueError("Invalid checkpoint horizon")
    if device != "reference":
        session = Session(
            data["program"],
            data["scenarios"],
            device=device,
            mode=mode,
            memory_limit=memory_limit,
            trace_cells=data["trace_cells"],
            trace_mode=data["trace_mode"],
            time_denominator=data["denominator"],
            **execution_options,
        )
    session.until = until
    session.results = data["results"]
    session._event_bound = max(r.events for r in session.results)
    if device == "reference":
        for engine, result in zip(session.interpreters, session.results):
            engine.state = result.state.copy()
            engine.time, engine.microstep = result.time, result.microstep
            engine.events, engine.status = result.events, result.status
            cal = result.calendar
            engine.cursor, engine.inputs = cal["cursor"], cal["inputs"].copy()
            engine.counts = [
                int(
                    (Fraction(t, session.denominator) - rational(c.phase))
                    / rational(c.period)
                )
                if t is not None
                else int(
                    (MAX * result.tick_seconds - rational(c.phase))
                    // rational(c.period)
                )
                + 1
                for c in session.program.clocks
                for t in (cal["clocks"][c.name],)
            ]
            engine.timers = {
                name: None
                if tag is None
                else (Fraction(tag[0], session.denominator), tag[1])
                for name, (tag, gen) in cal["timers"].items()
            }
            engine.generations = {
                name: gen for name, (tag, gen) in cal["timers"].items()
            }
    else:
        import numpy as np

        rows = []
        for result, state in zip(session.results, values):
            cal = result.calendar
            header = [
                {"horizon": 1, "budget": 2, "fault": 3, "complete": 4}[result.status],
                -1 if result.time is None else int(result.time * session.denominator),
                result.events,
                0,
                0,
                0,
                -1,
                -1,
                cal["cursor"],
                result.microstep,
                1,
                0,
            ]
            clocks = [
                cal["clocks"][c.name]
                if cal["clocks"][c.name] is not None
                else (1 << 63) - 1
                for c in session.program.clocks
            ]
            timers = []
            for t in session.program.timers:
                tag, gen = cal["timers"][t.name]
                timers.extend((*(tag if tag is not None else ((1 << 63) - 1, 0)), gen))
            rows.append(
                header
                + clocks
                + timers
                + [cal["inputs"][i.name] for i in session.program.inputs]
                + state
            )
        with session._stream_scope():
            session.states.assign(np.asarray(rows, dtype=np.int64).T.copy())
    return session
