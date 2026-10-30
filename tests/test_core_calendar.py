"""Calendar, rollback, replay and continuation contracts across common backends."""

from dataclasses import replace
from fractions import Fraction
import json

import pytest
import warp as wp

from warpsim.core import (Bits, BOOL, TIME, Cell, Input, Clock, Timer, Program,
                         Region, Schedule, Cancel, Store, Require, Observe, If,
                         Drive, const, read, inp, due, now, duration, microstep,
                         generation, op, lower, run_reference, run_native,
                         Session, Scenario)
from warpsim.core.serialization import Case

DEVICES = ["reference", "cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
U = Bits(8)


def execute(p, device, **options):
    expected = run_reference(p, **options)
    if device != "reference":
        actual = run_native(p, device=device, mode=wp.config.mode, **options)
        assert actual == expected
    return expected


def timer_program(body, *, schedule=(), initial="0", max_microstep=4):
    return Program("calendar", (Cell("x", U),), (),
                   (Region("fire", body, due("timer")), *schedule),
                   timers=(Timer("timer", initial),), max_microstep=max_microstep)


@pytest.mark.parametrize("device", DEVICES)
def test_timer_commit_rollback_and_captured_due(device):
    p = timer_program((Store("x", const(1, U)), Schedule("timer", duration("1/3"))),
        schedule=(Region("bad", (Cancel("timer"), Store("x", const(2, U)),
                                 Observe("unpublished"), Require(const(0, BOOL), "failure"))),))
    r = execute(p, device, until="2")
    assert r.state["x"] == (1,)
    assert r.calendar["timers"]["timer"] == ((1, 0), 1)
    assert r.fault.message == "failure"
    assert not any(x.kind == "observe" for x in r.records)
    p = timer_program((Cancel("timer"),), schedule=(
        Region("captured", (Observe("due", (due("timer"), generation("timer"), generation("timer", captured=True))),)),
        Region("valid_generation", (Store("x", const(9, U)),),
               op("eq", generation("timer"), generation("timer", captured=True))),
    ))
    r = execute(p, device, until="0")
    assert next(x.values for x in r.records if x.kind == "observe") == (1, 1, 0)
    assert r.state["x"] == (0,)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("body, fault", [
    ((Schedule("timer", duration("0")),), "event_order"),
    ((Schedule("timer", duration("0"), const(5)),), "microstep_limit"),
    ((Schedule("timer", duration("1")), Cancel("timer")), "conflict"),
    ((Schedule("timer", duration(str((1 << 62)-1))),), "overflow"),
])
def test_invalid_event_effects_roll_back(device, body, fault):
    p = timer_program((Store("x", const(7, U)), *body), initial="1")
    r = execute(p, device, until="1")
    assert r.fault.fault == fault and r.fault.target == "timer"
    assert r.state["x"] == (0,)
    # Dispatch consumed the source even though its handler rolled back.
    assert r.calendar["timers"]["timer"] == (None, 0)


def coincident():
    return Program("coincident", (Cell("x", U),), (Clock("clock", "1/3"),), (
        Region("capture", (Observe("sources", (due("clock"), due("timer"), inp("v", U))),)),
        Region("timer", (Store("x", op("add", read("x", U), const(1, U))),
                         If(op("eq", microstep(), const(0)),
                            (Schedule("timer", duration("0"), const(1)),),
                            (Schedule("timer", duration("1/3")),))), due("timer")),
    ), inputs=(Input("v", U),), timers=(Timer("timer", "0"),))


@pytest.mark.parametrize("device", DEVICES)
def test_exact_tags_coincidence_and_chunk_continuation(device):
    p = coincident()
    drives = (Drive("0", {"v": 3}), Drive("1/6", {"v": 7}), Drive("1/3", {"v": 9}))
    whole = execute(p, device, until="1", drives=drives, max_events=100)
    session = Session(p, (Scenario(drives),), device=device, mode=wp.config.mode)
    records = []
    while True:
        result = session.step(until="1", max_events=1)[0]
        records.extend(result.records)
        if result.status != "budget":
            break
    assert replace(result, records=tuple(records)) == whole
    observed = [r for r in records if r.kind == "observe"]
    assert [(r.time, r.microstep, r.values) for r in observed[:4]] == [
        (Fraction(0), 0, (1, 1, 3)), (Fraction(0), 1, (0, 1, 3)),
        (Fraction(1, 6), 0, (0, 0, 7)), (Fraction(1, 3), 0, (1, 1, 9)),
    ]


@pytest.mark.parametrize("device", DEVICES)
def test_zero_time_activity_is_bounded(device):
    p = timer_program((Schedule("timer", duration("0"), op("add", microstep(), const(1))),))
    r = execute(p, device, until="0")
    assert r.events == 5 and r.microstep == 4
    assert r.fault.fault == "microstep_limit"


@pytest.mark.parametrize("device", DEVICES)
def test_batch_and_trace_modes(device):
    p = coincident()
    scenarios = tuple(Scenario((Drive(str(Fraction(i, 7)), {"v": i}),), {"x": i}) for i in range(4))
    reference = Session(p, scenarios, device="reference").step(until="2", max_events=100)
    actual = Session(p, scenarios, device=device, mode=wp.config.mode).step(until="2", max_events=100)
    assert actual == reference
    for mode in ("observations", "summary"):
        compact = Session(p, scenarios, device=device, trace_mode=mode, mode=wp.config.mode).step(until="2", max_events=100)
        for a, b in zip(compact, reference):
            records = tuple(r for r in b.records if r.kind == "fault" or (mode == "observations" and r.kind == "observe"))
            assert a == replace(b, records=records)


@pytest.mark.parametrize("origin", DEVICES)
@pytest.mark.parametrize("target", DEVICES)
def test_portable_checkpoint_preserves_pending_microstep(origin, target, tmp_path):
    p = coincident()
    s = Session(p, device=origin, mode=wp.config.mode)
    first = s.step(until="2", max_events=1)[0]
    path = tmp_path / "state.json"
    s.checkpoint(path)
    restored = Session.restore(path, device=target, mode=wp.config.mode)
    continued = restored.step(until="2", max_events=100)[0]
    expected = run_reference(p, until="2", max_events=100)
    assert replace(continued, records=first.records + continued.records) == expected
    with pytest.raises(ValueError, match="nondecreasing"):
        restored.step(until="1")
    document = json.loads(path.read_text())
    document["data"]["denominator"] += 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="checksum"):
        Session.restore(path, device=target)


def test_closed_case_roundtrip_and_unavailable_backend(tmp_path):
    p = coincident()
    case = Case(p, "1", (Scenario((Drive("1/5", {"v": 7}),)),))
    path = tmp_path / "case.json"
    case.save(path)
    assert Case.load(path) == case
    assert Case.load(path).run(device="reference") == case.run(device="reference")
    with pytest.raises(Exception):
        case.run(device="cuda:987")
    with pytest.raises(ValueError, match="Unknown timer"):
        lower(timer_program((Cancel("missing"),)))
    with pytest.raises(ValueError, match="Duplicate symbol"):
        lower(replace(p, timers=(Timer("clock"),)))
