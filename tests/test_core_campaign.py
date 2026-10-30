"""Independent semantic checks for native resident campaign strategies.

Expected outcomes come from the interpreter and explicit assertions, never from
an alternate native scheduling configuration. No application model is imported.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest
import warp as wp

from warpsim.core import (
    BOOL,
    TIME,
    Bits,
    Cancel,
    Cell,
    Clock,
    Drive,
    Input,
    Observe,
    Program,
    Region,
    Scenario,
    Schedule,
    Session,
    Store,
    Table,
    Timer,
    const,
    due,
    duration,
    generation,
    inp,
    lookup,
    microstep,
    op,
    read,
)


NATIVE_DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
BYTE = Bits(8)
WORD = Bits(32)


def heterogeneous_batch(count=37):
    program = Program(
        "campaign_fixture",
        (Cell("identity", WORD), Cell("total", WORD)),
        (Clock("tick", "1/7"),),
        (
            Region(
                "accumulate",
                (
                    Observe("sample", (inp("index", BYTE), inp("halt", BOOL))),
                    Store(
                        "total",
                        op(
                            "add",
                            read("total", WORD),
                            lookup("weights", WORD, inp("index", BYTE)),
                        ),
                    ),
                ),
            ),
        ),
        inputs=(Input("index", BYTE), Input("halt", BOOL)),
        tables=(Table("weights", WORD, 2),),
        stop=inp("halt", BOOL),
    )
    schedules = (
        (Drive("0", {"halt": 1}),),
        (Drive("1/5", {"index": 2}),),
        (
            Drive("0", {"index": 0}),
            Drive("1/11", {"index": 1}),
            Drive("2/11", {"index": 0}),
            Drive("1/2", {"halt": 1}),
        ),
        (),
        (Drive("3/4", {"index": 1}),),
    )
    scenarios = tuple(
        Scenario(
            schedules[i % len(schedules)],
            {"identity": i, "total": i * 100},
            {"weights": [i + 1, i + 2]},
        )
        for i in range(count)
    )
    return program, scenarios


def make_session(program, scenarios=(Scenario(),), *, device="cpu", **options):
    return Session(
        program,
        scenarios,
        device=device,
        mode=wp.config.mode,
        trace_mode=options.pop("trace_mode", "summary"),
        **options,
    )


def assert_progress(progress, results):
    assert isinstance(progress, tuple)
    assert len(progress) == len(results)
    assert [p.scenario_id for p in progress] == list(range(len(results)))
    for p, r in zip(progress, results, strict=True):
        assert (p.status, p.time, p.events, p.microstep) == (
            r.status,
            r.time,
            r.events,
            r.microstep,
        )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        progress[0].events = -1


@pytest.mark.parametrize("device", NATIVE_DEVICES)
@pytest.mark.parametrize("compact", (False, True))
@pytest.mark.parametrize("quantum", (1, 3, 16))
@pytest.mark.parametrize("budget", (2, 7, 17))
def test_campaign_semantic_budget_is_independent_of_quantum(
    device, compact, quantum, budget
):
    program, scenarios = heterogeneous_batch()
    oracle = make_session(program, scenarios, device="reference")
    expected = oracle.run(until="1", max_events=budget)
    session = make_session(program, scenarios, device=device)
    actual = session.campaign(
        until="1", max_events=budget, quantum=quantum, compact=compact
    )
    assert actual == expected
    assert [r.state["identity"] for r in actual] == [
        (i,) for i in range(len(scenarios))
    ]
    assert_progress(session.status(), expected)
    assert session.inspect() == expected
    if budget == 17:
        assert [r.status for r in actual[:5]] == [
            "complete",
            "fault",
            "complete",
            "horizon",
            "horizon",
        ]
        assert actual[1].fault.fault == "bounds"
        assert actual[1].fault.index == 2
        assert actual[1].calendar["cursor"] == 1


@pytest.mark.parametrize("device", NATIVE_DEVICES)
@pytest.mark.parametrize("poll_interval", (1, 3, 20))
def test_compaction_polling_does_not_change_outcomes_or_active_storage_stride(
    device, poll_interval
):
    program, scenarios = heterogeneous_batch(67)
    expected = make_session(program, scenarios, device="reference").run(
        until="2", max_events=30
    )
    session = make_session(program, scenarios, device=device)
    assert (
        session.campaign(
            until="2",
            max_events=30,
            quantum=2,
            compact=True,
            poll_interval=poll_interval,
        )
        == expected
    )
    assert_progress(session.status(), expected)


@pytest.mark.parametrize("device", NATIVE_DEVICES)
def test_unseen_advances_keep_first_fault_and_historical_inspections(device):
    program, scenarios = heterogeneous_batch(5)
    session = make_session(program, scenarios, device=device)
    oracle = make_session(program, scenarios, device="reference")
    # The table fault happens during the second advance. Subsequent advances
    # must not overwrite its diagnostic buffer before anyone materializes it.
    for budget in (1, 2, 1, 3):
        assert session.advance(until="1", max_events=budget) is None
        expected = oracle.step(until="1", max_events=budget)
    assert session.inspect() == expected
    assert expected[1].fault is not None and expected[1].fault.time == Fraction(1, 5)
    frozen = session.inspect()
    snapshot = deepcopy(frozen)
    assert_progress(session.status(), expected)
    assert session.advance(until="2", max_events=4) is None
    later = oracle.step(until="2", max_events=4)
    assert session.inspect() == later
    assert frozen == snapshot
    assert frozen[3].events < later[3].events
    assert frozen[1].fault == later[1].fault
    # Returning to the eager public API uses exactly the resident frontier.
    assert session.step(until="3", max_events=9) == oracle.step(until="3", max_events=9)


def cancellation_program():
    return Program(
        "campaign_calendar",
        (Cell("seen", WORD), Cell("captured", TIME), Cell("current", TIME)),
        (Clock("clock", "1", "1"),),
        (
            Region("cancel", (Cancel("alarm"),), due("clock")),
            Region(
                "dispatch",
                (
                    Store("seen", op("add", read("seen", WORD), const(1, WORD))),
                    Store("captured", generation("alarm", captured=True)),
                    Store("current", generation("alarm")),
                    Schedule("chain", duration("0"), const(1)),
                ),
                due("alarm"),
            ),
            Region(
                "chain",
                (
                    Store("seen", op("add", read("seen", WORD), const(1, WORD))),
                    Schedule("chain", duration("0"), op("add", microstep(), const(1))),
                ),
                due("chain"),
            ),
        ),
        timers=(Timer("alarm", "1"), Timer("chain")),
        max_microstep=3,
    )


@pytest.mark.parametrize("device", NATIVE_DEVICES)
def test_resident_cancellation_dispatch_and_microstep_fault_are_exact(device):
    program = cancellation_program()
    session = make_session(program, device=device)
    oracle = make_session(program, device="reference")
    session.advance(until="2", max_events=1)
    expected = oracle.step(until="2", max_events=1)
    assert session.inspect() == expected
    assert expected[0].state == {"seen": (1,), "captured": (0,), "current": (1,)}
    assert expected[0].calendar["timers"]["chain"] == ((1, 1), 1)
    for _ in range(5):
        session.advance(until="2", max_events=1)
        expected = oracle.step(until="2", max_events=1)
    assert session.inspect() == expected
    result = expected[0]
    assert result.status == "fault" and result.microstep == 3
    assert result.fault.fault == "microstep_limit"
    assert result.state["seen"] == (3,)
    assert result.calendar["timers"]["chain"] == (None, 3)


@pytest.mark.parametrize("origin", NATIVE_DEVICES)
@pytest.mark.parametrize("target", NATIVE_DEVICES)
@pytest.mark.parametrize("before_fault", (True, False))
def test_checkpoint_materializes_unseen_resident_frontier(
    origin, target, before_fault, tmp_path
):
    program, scenarios = heterogeneous_batch(7)
    session = make_session(program, scenarios, device=origin)
    oracle = make_session(program, scenarios, device="reference")
    budget = 1 if before_fault else 5
    session.advance(until="1", max_events=budget)
    expected = oracle.step(until="1", max_events=budget)
    path = tmp_path / "resident.json"
    session.checkpoint(path)
    restored = Session.restore(
        path,
        device=target,
        mode=wp.config.mode,
        cpu_threads=2,
        cpu_schedule="dynamic",
        cpu_chunk=1,
    )
    assert restored.inspect() == expected
    assert_progress(restored.status(), expected)
    for quantum in (1, 3, 8):
        assert restored.campaign(
            until="2", max_events=11, quantum=quantum, compact=True
        ) == oracle.run(
            until="2",
            max_events=11,
        )


@pytest.mark.parametrize("trace_mode", ("full", "observations", "summary"))
@pytest.mark.parametrize("reuse_buffers", (False, True))
@pytest.mark.parametrize("device", NATIVE_DEVICES)
def test_eager_step_reuse_preserves_records_and_returned_snapshots(
    device, reuse_buffers, trace_mode
):
    program, scenarios = heterogeneous_batch(7)
    options = dict(trace_mode=trace_mode, trace_cells=("total",))
    session = make_session(
        program, scenarios, device=device, reuse_buffers=reuse_buffers, **options
    )
    oracle = make_session(program, scenarios, device="reference", **options)
    history = []
    for horizon, budget in (("1/7", 1), ("1/2", 8), ("1", 2), ("2", 14)):
        actual = session.step(until=horizon, max_events=budget)
        assert actual == oracle.step(until=horizon, max_events=budget)
        history.append((actual, deepcopy(actual)))
    assert all(actual == snapshot for actual, snapshot in history)


@pytest.mark.parametrize("schedule", ("static", "dynamic"))
@pytest.mark.parametrize("threads", (1, 2, 4))
@pytest.mark.parametrize("chunk", (1, 3))
def test_native_cpu_parallel_scheduling_matches_reference(schedule, threads, chunk):
    program, scenarios = heterogeneous_batch(67)
    session = make_session(
        program,
        scenarios,
        cpu_threads=threads,
        cpu_schedule=schedule,
        cpu_chunk=chunk,
    )
    expected = make_session(program, scenarios, device="reference").run(
        until="2", max_events=40
    )
    assert (
        session.campaign(until="2", max_events=40, quantum=3, compact=True) == expected
    )
    assert_progress(session.status(), expected)


@pytest.mark.parametrize("trace_mode", ("full", "observations"))
@pytest.mark.parametrize("device", NATIVE_DEVICES)
def test_resident_api_rejects_history_dropping_modes_before_advancing(
    device, trace_mode
):
    program, scenarios = heterogeneous_batch(2)
    session = make_session(program, scenarios, device=device, trace_mode=trace_mode)
    with pytest.raises(ValueError):
        session.advance(until="1", max_events=4)
    with pytest.raises(ValueError):
        session.campaign(until="1", max_events=4, quantum=1)
    assert session.step(until="1", max_events=4) == make_session(
        program,
        scenarios,
        device="reference",
        trace_mode=trace_mode,
    ).step(until="1", max_events=4)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is required")
@pytest.mark.parametrize("strategy", ("scalar", "warp", "block"))
@pytest.mark.parametrize("block_size", (32, 128))
@pytest.mark.parametrize("with_map", (False, True))
def test_cuda_strategy_preserves_serial_calendar_semantics(
    strategy, block_size, with_map, tmp_path
):
    program = cancellation_program()
    if with_map:
        program = replace(
            program,
            cells=program.cells
            + (Cell("map_source", BYTE, 33), Cell("mapped", WORD, 33)),
            schedule=(
                Region(
                    "map",
                    tuple(
                        Store(
                            "mapped",
                            op(
                                "add",
                                op(
                                    "cast",
                                    read("map_source", BYTE, const(i)),
                                    type=WORD,
                                ),
                                const(4, WORD),
                            ),
                            const(i),
                        )
                        for i in range(33)
                    ),
                ),
                *program.schedule,
            ),
        )
    session = make_session(
        program, device="cuda:0", strategy=strategy, block_size=block_size
    )
    if with_map and strategy != "scalar":
        assert session.cooperative_regions
    oracle = make_session(program, device="reference")
    session.advance(until="2", max_events=1)
    oracle.step(until="2", max_events=1)
    path = tmp_path / "strategy.json"
    session.checkpoint(path)
    restored = Session.restore(
        path,
        device="cuda:0",
        mode=wp.config.mode,
        strategy=strategy,
        block_size=block_size,
    )
    assert restored.campaign(
        until="2", max_events=8, quantum=1, compact=True
    ) == oracle.run(
        until="2",
        max_events=8,
    )


@pytest.mark.parametrize("device", NATIVE_DEVICES)
def test_resident_zero_event_horizon_and_compacted_campaign_restart(device):
    program = Program(
        "campaign_horizon",
        (Cell("count", WORD), Cell("limit", WORD)),
        (Clock("tick", "1/3", "1"),),
        (
            Region(
                "count",
                (Store("count", op("add", read("count", WORD), const(1, WORD))),),
            ),
        ),
        stop=op("ge", read("count", WORD), read("limit", WORD)),
    )
    scenarios = tuple(
        Scenario(initial={"limit": value}) for value in (1, 3, 10, 2, 100)
    )
    session = make_session(program, scenarios, device=device)
    oracle = make_session(program, scenarios, device="reference")
    session.advance(until="1/2", max_events=9)
    expected = oracle.step(until="1/2", max_events=9)
    assert all(
        r.time is None and r.events == 0 and r.status == "horizon" for r in expected
    )
    assert session.inspect() == expected
    assert_progress(session.status(), expected)
    for horizon, budget, quantum in (("1", 3, 1), ("3", 2, 1), ("4", 11, 4)):
        assert session.campaign(
            until=horizon,
            max_events=budget,
            quantum=quantum,
            compact=True,
        ) == oracle.run(until=horizon, max_events=budget)


@pytest.mark.parametrize("device", NATIVE_DEVICES)
@pytest.mark.parametrize("reuse_buffers", (False, True))
def test_resident_advances_avoid_inspection_and_reuse_is_measurable(
    device, reuse_buffers
):
    program, scenarios = heterogeneous_batch(5)
    session = make_session(
        program, scenarios, device=device, reuse_buffers=reuse_buffers, profile=True
    )
    oracle = make_session(program, scenarios, device="reference")
    for _ in range(3):
        session.advance(until="1", max_events=1)
        expected = oracle.step(until="1", max_events=1)
    assert session.metrics["launches"] == 3
    assert session.metrics["trace_allocations"] == (1 if reuse_buffers else 3)
    assert session.metrics["host_readback_bytes"] == 0
    assert session.metrics["decode_seconds"] == 0
    assert_progress(session.status(), expected)
    compact_bytes = session.metrics["host_readback_bytes"]
    assert compact_bytes > 0
    assert session.metrics["decode_seconds"] == 0
    assert session.inspect() == expected
    assert session.metrics["host_readback_bytes"] - compact_bytes > compact_bytes
    assert session.metrics["native_execution_seconds"] > 0
