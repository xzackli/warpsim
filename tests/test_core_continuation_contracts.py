"""Independent continuation checks using declared expectations and adversarial data."""

from dataclasses import replace
from fractions import Fraction
import hashlib
import json

import pytest
import warp as wp

from warpsim.core import (
    BOOL, TIME, Bits, Cell, Input, Table, Clock, Timer, Program, Region,
    Repeat, Schedule, Cancel, Store, Require, Observe, Drive, const, read,
    inp, due, now, duration, generation, op, lookup, Session, Scenario,
    run_reference, run_native,
)
from warpsim.core.serialization import Case, canonical, decode, encode
from warpsim.core.lower import MAX


DEVICES = ["reference", "cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
BYTE = Bits(8)
WORD = Bits(16)


def compare(program, device, **options):
    expected = run_reference(program, **options)
    if device != "reference":
        assert run_native(program, device=device, mode=wp.config.mode, **options) == expected
    return expected


@pytest.mark.parametrize("device", DEVICES)
def test_empty_calendar_and_horizon_before_first_event(device, tmp_path):
    empty = Program("empty", (Cell("x", BYTE, initial=(17,)),), (), ())
    session = Session(empty, device=device, mode=wp.config.mode)
    for horizon in ("0", "1/7", "10"):
        result = session.step(until=horizon, max_events=1)[0]
        assert result.status == "horizon"
        assert result.events == 0 and result.time is None and result.records == ()
        assert result.state == {"x": (17,)}
    path = tmp_path / "empty.json"
    session.checkpoint(path)
    assert Session.restore(path, device=device, mode=wp.config.mode).step(until="11")[0] == result

    periodic = replace(empty, name="boundary", clocks=(Clock("clock", "1/3", "1/3"),),
                       schedule=(Region("count", (Store("x", op("add", read("x", BYTE), const(1, BYTE))),)),))
    session = Session(periodic, device=device, mode=wp.config.mode)
    before = session.step(until="333/1000", max_events=1)[0]
    assert before.events == 0 and before.time is None
    assert before.calendar["clocks"] == {"clock": 1}
    session.checkpoint(path)
    restored = Session.restore(path, device=device, mode=wp.config.mode)
    at_boundary = restored.step(until="1/3", max_events=1)[0]
    assert at_boundary == compare(periodic, device, until="1/3", max_events=1)
    assert at_boundary.status == "horizon" and at_boundary.events == 1
    after = restored.step(until="1/2", max_events=1)[0]
    assert after == replace(at_boundary, records=())


@pytest.mark.parametrize("device", DEVICES)
def test_maximum_lattice_tick_allows_fractional_horizon_checkpoint(device, tmp_path):
    p = Program("maximum_horizon", (Cell("x", BYTE),), (Clock("clock", "1", str(MAX - 1)),), (
        Region("count", (Store("x", op("add", read("x", BYTE), const(1, BYTE))),)),
    ))
    session = Session(p, device=device, mode=wp.config.mode)
    horizon = str(Fraction(2 * MAX + 1, 2))
    expected = session.step(until=horizon)
    assert expected[0].events == 2 and expected[0].time == Fraction(MAX)
    assert expected[0].state["x"] == (2,) and expected[0].calendar["clocks"] == {"clock": None}
    path = tmp_path / "maximum.json"
    session.checkpoint(path)
    resumed = Session.restore(path, device=device, mode=wp.config.mode).step(until=horizon)
    assert resumed == (replace(expected[0], records=()),)


@pytest.mark.parametrize("device", DEVICES)
def test_replacement_and_cancel_only_publish_committed_generations(device):
    p = Program("timer_replacements", (Cell("x", BYTE),), (), (
        Region("first", (Schedule("alarm", duration("1/2")),), due("seed")),
        Region("replace", (Schedule("alarm", duration("1/3")),), due("seed")),
        Region("cancel", (Cancel("alarm"),), due("seed")),
        Region("last", (Schedule("alarm", duration("2/3")),), due("seed")),
        Region("seen", (Observe("generation", (due("alarm"), generation("alarm"),
                                               generation("alarm", captured=True))),)),
        Region("fire", (Store("x", op("add", read("x", BYTE), const(1, BYTE))),), due("alarm")),
    ), timers=(Timer("seed", "0"), Timer("alarm", "10")))
    result = compare(p, device, until="1")
    assert result.events == 2 and result.time == Fraction(2, 3)
    assert result.state == {"x": (1,)}
    assert result.calendar["timers"] == {"seed": (None, 0), "alarm": (None, 4)}
    assert [(r.time, r.values) for r in result.records if r.kind == "observe"] == [
        (Fraction(0), (0, 4, 0)), (Fraction(2, 3), (1, 4, 4)),
    ]


@pytest.mark.parametrize("device", DEVICES)
def test_repeat_failure_preserves_prior_timer_commits_and_fault_iteration(device):
    p = Program("repeat_timer_fault", (Cell("x", BYTE),), (), (
        Repeat("rounds", op("lt", read("x", BYTE), const(3, BYTE)), 3, (
            Region("commit", (Store("x", op("add", read("x", BYTE), const(1, BYTE))),
                              Schedule("alarm", duration("1/3")))),
            Region("rollback", (Cancel("alarm"), Observe("tentative"),
                                Require(op("lt", read("x", BYTE), const(2, BYTE)), "second round"))),
        )),
    ), timers=(Timer("seed", "0"), Timer("alarm")))
    result = compare(p, device, until="1")
    assert result.state == {"x": (2,)}
    assert result.fault.iteration == 1 and result.fault.region == "rollback"
    assert result.calendar["timers"]["alarm"] == ((1, 0), 3)
    observed = [r for r in result.records if r.kind == "observe"]
    assert len(observed) == 1 and observed[0].iteration == 0
    assert [r.iteration for r in result.records if r.region == "commit"] == [0, 1]


def heterogeneous_case():
    p = Program("continuation_lanes", (Cell("x", WORD), Cell("unused", BYTE)),
                (Clock("clock", "1/7"),), (
                    Region("accumulate", (
                        Observe("input", (now(), inp("index", BYTE), inp("halt", BOOL))),
                        Store("x", op("add", read("x", WORD), lookup("data", WORD, inp("index", BYTE)))),
                    )),
                ), inputs=(Input("index", BYTE), Input("halt", BOOL)),
                tables=(Table("data", WORD, 2),), stop=inp("halt", BOOL))
    scenarios = (
        Scenario((Drive("0", {"halt": 1}),), {"x": 10}, {"data": [1, 2]}),
        Scenario((Drive("1/5", {"index": 2}),), {"x": 20}, {"data": [3, 4]}),
        Scenario((Drive("0", {"index": 0}), Drive("1/11", {"index": 1}),
                  Drive("2/11", {"index": 0}), Drive("1/2", {"halt": 1})),
                 {"x": 30}, {"data": [5, 6]}),
        Scenario((), {"x": 40}, {"data": [7, 8]}),
    )
    return p, scenarios


@pytest.mark.parametrize("origin", DEVICES)
@pytest.mark.parametrize("target", DEVICES)
@pytest.mark.parametrize("trace_mode", ("full", "observations", "summary"))
def test_mixed_terminal_lanes_tables_inputs_and_trace_survive_restore(origin, target, trace_mode, tmp_path):
    p, scenarios = heterogeneous_case()
    options = dict(trace_mode=trace_mode, trace_cells=("x",), mode=wp.config.mode)
    original = Session(p, scenarios, device=origin, **options)
    expected = Session(p, scenarios, device="reference", **options)
    for _ in range(2):
        assert original.step(until="1", max_events=1) == expected.step(until="1", max_events=1)
    path = tmp_path / "mixed.json"
    original.checkpoint(path)
    restored = Session.restore(path, device=target, mode=wp.config.mode)
    # Checkpoint history is intentionally compact. Compare both executions
    # from the same saved frontier, including previously terminal lanes.
    oracle = Session.restore(path, device="reference", mode=wp.config.mode)
    for _ in range(10):
        actual = restored.step(until="1", max_events=1)
        assert actual == oracle.step(until="1", max_events=1)
        assert all(set(record.state) == {"x"} for result in actual for record in result.records)
    assert [r.status for r in actual] == ["complete", "fault", "complete", "horizon"]
    assert actual[0].state["x"] == (11,) and actual[0].events == 1
    assert actual[1].state["x"] == (26,) and actual[1].events == 3
    assert actual[1].calendar["cursor"] == 1 and actual[1].calendar["inputs"]["index"] == 2
    assert actual[1].fault.fault == "bounds" and actual[1].fault.target == "data"
    assert actual[2].calendar["cursor"] == 4 and actual[2].time == Fraction(1, 2)
    assert actual[3].state["x"] == (96,) and actual[3].events == 8
    assert actual[3].calendar["cursor"] == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trace_mode", ("full", "observations", "summary"))
def test_fault_metadata_is_independent_of_empty_trace_selection(device, trace_mode):
    p, scenarios = heterogeneous_case()
    results = Session(p, scenarios, device=device, mode=wp.config.mode,
                      trace_mode=trace_mode, trace_cells=()).step(until="1")
    assert all(r.state == {} for result in results for r in result.records)
    fault = results[1].fault
    assert (fault.time, fault.microstep, fault.region, fault.fault, fault.target, fault.index) == (
        Fraction(1, 5), 0, "accumulate", "bounds", "data", 2,
    )
    assert results[1].state["x"] == (26,)
    if trace_mode == "summary":
        assert [len(r.records) for r in results] == [0, 1, 0, 0]


def rewrite_checkpoint(path, transform):
    document = json.loads(path.read_text())
    data = decode(document["data"])
    transform(data)
    document["data"] = encode(data)
    document["sha256"] = hashlib.sha256(canonical(document["data"]).encode()).hexdigest()
    path.write_text(json.dumps(document))


@pytest.mark.parametrize("corruption", ("skip_clock", "lose_clock", "skip_input", "lose_frontier", "zero_events", "past_horizon"))
def test_checkpoint_rejects_inconsistent_frontier_even_with_matching_checksum(corruption, tmp_path):
    p, scenarios = heterogeneous_case()
    session = Session(p, (scenarios[2],), device="reference")
    session.step(until="1", max_events=1)
    path = tmp_path / "bad.json"
    session.checkpoint(path)

    def corrupt(data):
        result = data["results"][0]
        if corruption == "skip_clock":
            result.calendar["clocks"]["clock"] += session.denominator // 7
        elif corruption == "lose_clock":
            result.calendar["clocks"]["clock"] = None
        elif corruption == "skip_input":
            result.calendar["cursor"] = 2
            result.calendar["inputs"]["index"] = 1
        elif corruption == "lose_frontier":
            result = replace(result, time=None)
        elif corruption == "zero_events":
            result = replace(result, events=0)
        elif corruption == "past_horizon":
            data["until"] = "0"
            result = replace(result, time=Fraction(1, 11))
        data["results"] = (result,)

    rewrite_checkpoint(path, corrupt)
    with pytest.raises(ValueError):
        Session.restore(path, device="reference")


def test_case_uses_closed_constructors_and_rejects_extra_fields(tmp_path):
    p, scenarios = heterogeneous_case()
    case = Case(p, "1", scenarios)
    path = tmp_path / "case.json"
    case.save(path)
    assert Case.load(path) == case
    document = json.loads(path.read_text())
    document["data"]["$type"] = "__import__"
    document["sha256"] = hashlib.sha256(canonical(document["data"]).encode()).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="constructor"):
        Case.load(path)
    with pytest.raises(ValueError, match="constructor"):
        decode({"$type": "Bits", "width": 8, "signed": False, "code": "unexpected"})


@pytest.mark.parametrize("corruption", ("fault_without_diagnostic", "nonfault_history"))
def test_checkpoint_rejects_invalid_diagnostics_before_native_allocation(corruption, tmp_path, monkeypatch):
    p, scenarios = heterogeneous_case()
    session = Session(p, (scenarios[3],), device="reference")
    result = session.step(until="0")[0]
    path = tmp_path / "diagnostics.json"
    session.checkpoint(path)

    def corrupt(data):
        saved = data["results"][0]
        data["results"] = (replace(saved, status="fault") if corruption == "fault_without_diagnostic"
                            else replace(saved, records=result.records),)

    rewrite_checkpoint(path, corrupt)
    def prohibit_native(self):
        pytest.fail("Invalid checkpoint reached native initialization")
    monkeypatch.setattr(Session, "_initialize_native", prohibit_native)
    with pytest.raises(ValueError):
        Session.restore(path, device="cpu")


@pytest.mark.parametrize("device", DEVICES)
def test_large_campaign_uses_chunks_and_preserves_terminal_lanes(device):
    p = Program("large_campaign", (Cell("x", TIME),), (Clock("clock"),), (
        Region("count", (Store("x", op("add", read("x", TIME), const(1))),)),
    ), stop=op("eq", read("x", TIME), const(2)))
    scenarios = (Scenario(initial={"x": 0}), Scenario(initial={"x": 3}), Scenario(initial={"x": MAX}))
    session = Session(p, scenarios, device=device, trace_mode="summary", mode=wp.config.mode)
    result = session.run(until="200000", max_events=100005)
    assert [r.status for r in result] == ["complete", "budget", "fault"]
    assert [r.events for r in result] == [2, 100005, 1]
    assert [r.state["x"] for r in result] == [(2,), (100008,), (MAX,)]
    assert [len(r.records) for r in result] == [0, 0, 1]
    assert result[2].fault.fault == "overflow"
    continuation = session.run(until="200000", max_events=3)
    assert continuation[0] == result[0] and continuation[2] == result[2]
    assert continuation[1].events == 100008 and continuation[1].time == Fraction(100007)
    assert continuation[1].state["x"] == (100011,)


@pytest.mark.parametrize("device", DEVICES)
def test_large_campaign_aggregates_sparse_observations_across_chunks(device):
    p = Program("large_campaign_observations", (Cell("x", TIME),), (Clock("clock"),), (
        Region("observe", (Observe("boundary", (read("x", TIME),)),),
               op("or", op("eq", read("x", TIME), const(0)),
                  op("eq", read("x", TIME), const(100000)))),
        Region("count", (Store("x", op("add", read("x", TIME), const(1))),)),
    ), stop=op("eq", read("x", TIME), const(100002)))
    session = Session(p, device=device, trace_mode="observations", mode=wp.config.mode)
    result = session.run(until="200000", max_events=100005)[0]
    assert result.status == "complete" and result.events == 100002
    assert [(r.time, r.values) for r in result.records] == [(Fraction(0), (0,)), (Fraction(100000), (100000,))]
