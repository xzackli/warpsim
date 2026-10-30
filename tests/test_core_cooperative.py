"""Cooperative lowering must preserve the ordinary transaction language.

These declarations use no GPU or application opcode. All expected results are
computed by the independent interpreter before selecting an execution strategy.
"""

from dataclasses import replace

import pytest
import warp as wp

from warpsim.core import (
    BOOL,
    Bits,
    Cell,
    Clock,
    Drive,
    Input,
    Observe,
    Program,
    Region,
    Require,
    Scenario,
    Schedule,
    Session,
    Store,
    Table,
    Timer,
    const,
    duration,
    inp,
    lookup,
    microstep,
    op,
    read,
)
from warpsim.core.codegen import Emitter
from warpsim.core.lower import lower


BYTE = Bits(8)
WORD = Bits(32)
CUDA = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is required")
STRATEGIES = ("scalar", "warp", "block")


def widen(value):
    return op("cast", value, type=WORD)


def map_program(style="checked_add", size=33):
    def expression(i):
        source = read("source", BYTE, const(i))
        if style == "checked_add":
            return op("add", widen(source), const(300, WORD))
        if style == "checked_mul":
            return op("mul", widen(source), const(17, WORD))
        if style in ("wrap", "saturate"):
            return widen(op("add", source, const(200, BYTE), policy=style))
        if style == "compare":
            return widen(op("lt", source, lookup("weights", BYTE, const(i))))
        if style == "stencil":
            return op(
                "add",
                op("add", widen(source), widen(read("source", BYTE, const(i + 1)))),
                widen(lookup("weights", BYTE, const(i + 2))),
            )
        raise AssertionError(style)

    return Program(
        "cooperative_map",
        (
            Cell("source", BYTE, size + 2),
            Cell("output", WORD, size),
            Cell("prefix", WORD),
        ),
        (Clock("tick", "1/3"),),
        (
            Region(
                "prefix",
                (Store("prefix", op("add", read("prefix", WORD), const(1, WORD))),),
            ),
            Region(
                "map",
                tuple(Store("output", expression(i), const(i)) for i in range(size)),
            ),
            Region(
                "inspect",
                (
                    Observe(
                        "ends",
                        (read("output", WORD), read("output", WORD, const(size - 1))),
                    ),
                ),
            ),
        ),
        tables=(Table("weights", BYTE, size + 2),),
    )


def scenarios_for(size, count):
    return tuple(
        Scenario(
            initial={"source": [(i * 23 + lane * 41) % 256 for i in range(size + 2)]},
            tables={"weights": [(i * 17 + lane * 31) % 256 for i in range(size + 2)]},
        )
        for lane in range(count)
    )


def session_for(
    program,
    scenarios=(Scenario(),),
    *,
    device="reference",
    strategy="scalar",
    trace_mode="summary",
):
    return Session(
        program,
        scenarios,
        device=device,
        strategy=strategy,
        trace_mode=trace_mode,
        mode=wp.config.mode,
    )


@CUDA
@pytest.mark.parametrize(
    "style", ("checked_add", "checked_mul", "wrap", "saturate", "compare", "stencil")
)
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("trace_mode", ("summary", "full"))
def test_regular_maps_match_scalar_reference_and_preserve_trace_order(
    style, strategy, trace_mode
):
    program = map_program(style)
    scenarios = scenarios_for(33, 3)
    expected = session_for(program, scenarios, trace_mode=trace_mode).run(
        until="1", max_events=8
    )
    session = session_for(
        program, scenarios, device="cuda:0", strategy=strategy, trace_mode=trace_mode
    )
    if strategy != "scalar":
        assert isinstance(session.cooperative_regions, tuple)
        assert session.cooperative_regions
    actual = session.run(until="1", max_events=8)
    assert actual == expected
    assert all(r.state["prefix"] == (4,) for r in actual)


@CUDA
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("trace_mode", ("summary", "full"))
def test_partial_lane_stencil_and_heterogeneous_batch(strategy, trace_mode):
    program = map_program("stencil", 65)
    scenarios = scenarios_for(65, 37)
    expected = session_for(program, scenarios, trace_mode=trace_mode).run(
        until="2/3", max_events=5
    )
    session = session_for(
        program, scenarios, device="cuda:0", strategy=strategy, trace_mode=trace_mode
    )
    assert session.run(until="2/3", max_events=5) == expected
    assert len({r.state["output"] for r in expected}) == 37


@CUDA
@pytest.mark.parametrize("strategy", ("warp", "block"))
def test_cooperative_in_place_neighbor_map_reads_transaction_entry_state(strategy):
    size = 65
    program = Program(
        "cooperative_entry_view",
        (Cell("values", BYTE, size + 1),),
        (Clock("tick"),),
        (
            Region(
                "shift",
                tuple(
                    Store("values", read("values", BYTE, const(i + 1)), const(i))
                    for i in range(size)
                ),
            ),
        ),
    )
    scenarios = (Scenario(initial={"values": list(range(size + 1))}),)
    oracle = session_for(program, scenarios)
    session = session_for(program, scenarios, device="cuda:0", strategy=strategy)
    assert session.cooperative_regions
    first = session.step(until="0", max_events=1)
    assert first == oracle.step(until="0", max_events=1)
    assert first[0].state["values"] == tuple(range(1, size + 1)) + (size,)
    assert session.campaign(
        until="3", max_events=5, quantum=1, compact=True
    ) == oracle.run(until="3", max_events=5)


@CUDA
@pytest.mark.parametrize("strategy", ("warp", "block"))
@pytest.mark.parametrize(
    "failure", ("requirement", "later_alias", "earlier_alias", "earlier_bounds")
)
def test_cooperative_candidates_preserve_rollback_prefix_and_first_fault(
    strategy, failure
):
    program = map_program()
    map_region = program.schedule[1]
    if failure == "requirement":
        body = map_region.body + (
            Require(const(0, BOOL), "late failure", (read("prefix", WORD),)),
        )
    elif failure == "later_alias":
        body = map_region.body + (Store("output", const(9, WORD), const(7)),)
    else:
        body = (Store("output", const(9, WORD), inp("alias", WORD)),) + map_region.body
    program = replace(
        program,
        schedule=(
            program.schedule[0],
            replace(map_region, body=body),
            program.schedule[2],
        ),
        inputs=(Input("alias", WORD, 99 if failure == "earlier_bounds" else 7),),
    )
    scenarios = scenarios_for(33, 3)
    expected = session_for(program, scenarios, trace_mode="full").run(
        until="1", max_events=8
    )
    session = session_for(
        program, scenarios, device="cuda:0", strategy=strategy, trace_mode="full"
    )
    assert session.run(until="1", max_events=8) == expected
    for result in expected:
        assert result.status == "fault" and result.events == 1
        assert result.state["prefix"] == (1,)
        assert result.state["output"] == (0,) * 33
        assert result.fault.region == "map"
        assert not any(r.kind == "observe" for r in result.records)
        assert (
            result.fault.fault
            == {
                "requirement": "requirement",
                "later_alias": "conflict",
                "earlier_alias": "conflict",
                "earlier_bounds": "bounds",
            }[failure]
        )


def unsafe_program(kind, size=33):
    def expression(i):
        source = read("source", BYTE, const(i))
        if kind == "overflow":
            return op("add", source, const(250, BYTE))
        if kind == "division":
            return op("div", source, inp("divisor", BYTE))
        if kind == "bounds":
            return read("source", BYTE, inp("index", WORD))
        if kind == "lazy":
            return op(
                "select",
                inp("enabled", BOOL),
                source,
                read("source", BYTE, const(size + 3)),
            )
        raise AssertionError(kind)

    return Program(
        "cooperative_scalar_fallback",
        (Cell("source", BYTE, size), Cell("output", BYTE, size)),
        (Clock("tick"),),
        (
            Region(
                "map",
                tuple(Store("output", expression(i), const(i)) for i in range(size)),
            ),
        ),
        inputs=(
            Input("divisor", BYTE),
            Input("index", WORD, size + 3),
            Input("enabled", BOOL, 1),
        ),
    )


@CUDA
@pytest.mark.parametrize("strategy", ("warp", "block"))
@pytest.mark.parametrize("kind", ("overflow", "division", "bounds", "lazy"))
def test_potential_faults_and_lazy_branches_stay_in_scalar_order(strategy, kind):
    program = unsafe_program(kind)
    scenarios = (Scenario(initial={"source": [1, 2, 3, 254] + [17] * 29}),)
    expected = session_for(program, scenarios, trace_mode="full").run(
        until="1", max_events=4
    )
    session = session_for(
        program, scenarios, device="cuda:0", strategy=strategy, trace_mode="full"
    )
    assert session.cooperative_regions == ()
    assert session.run(until="1", max_events=4) == expected
    if kind == "lazy":
        assert expected[0].status == "horizon"
        assert expected[0].state["output"] == expected[0].state["source"]
    else:
        assert expected[0].status == "fault"
        assert expected[0].state["output"] == (0,) * 33


@CUDA
@pytest.mark.parametrize("strategy", ("warp", "block"))
def test_guard_fault_and_disabled_map_do_not_execute_cooperative_candidates(strategy):
    program = map_program()
    program = replace(
        program,
        inputs=(Input("guard_index", WORD),),
        schedule=(
            program.schedule[0],
            replace(
                program.schedule[1],
                guard=op(
                    "eq",
                    lookup("weights", BYTE, inp("guard_index", WORD)),
                    const(255, BYTE),
                ),
            ),
            program.schedule[2],
        ),
    )
    base = scenarios_for(33, 3)
    scenarios = tuple(
        replace(
            s,
            drives=(Drive("0", {"guard_index": 99 if i == 1 else 0}),),
            tables={"weights": [255, *s.tables["weights"][1:]]} if i == 2 else s.tables,
        )
        for i, s in enumerate(base)
    )
    expected = session_for(program, scenarios, trace_mode="full").run(
        until="1", max_events=8
    )
    session = session_for(
        program, scenarios, device="cuda:0", strategy=strategy, trace_mode="full"
    )
    assert session.run(until="1", max_events=8) == expected
    assert [r.status for r in expected] == ["horizon", "fault", "horizon"]
    assert all(r.state["output"] == (0,) * 33 for r in expected[:2])
    assert all(value >= 300 for value in expected[2].state["output"])


@CUDA
@pytest.mark.parametrize("origin", ("cpu", "scalar", "warp", "block"))
@pytest.mark.parametrize("target", ("cpu", "scalar", "warp", "block"))
def test_cooperative_checkpoint_is_portable_across_strategies(origin, target, tmp_path):
    program = map_program("stencil")
    scenarios = scenarios_for(33, 3)
    session = session_for(
        program,
        scenarios,
        device="cpu" if origin == "cpu" else "cuda:0",
        strategy="scalar" if origin == "cpu" else origin,
    )
    oracle = session_for(program, scenarios)
    session.advance(until="1", max_events=2)
    expected = oracle.step(until="1", max_events=2)
    path = tmp_path / "cooperative.json"
    session.checkpoint(path)
    restored = Session.restore(
        path,
        device="cpu" if target == "cpu" else "cuda:0",
        strategy="scalar" if target == "cpu" else target,
        mode=wp.config.mode,
    )
    assert restored.inspect() == expected
    assert restored.campaign(
        until="2", max_events=10, quantum=2, compact=True
    ) == oracle.run(until="2", max_events=10)


def test_cooperative_emission_uses_execution_ir_after_semantic_bodies_are_removed():
    program = map_program("stencil")
    for strategy in ("warp", "block"):
        emitter = Emitter(lower(program), strategy=strategy)
        emitter.p = replace(emitter.p, captures=None, schedule=None, stop=None)
        assert (
            emitter.generate() == Emitter(lower(program), strategy=strategy).generate()
        )


@CUDA
@pytest.mark.parametrize("strategy", ("warp", "block"))
@pytest.mark.parametrize("policy", ("wrap", "saturate"))
def test_signed_conversion_and_reverse_destination_stride(strategy, policy):
    signed = Bits(8, True)
    size = 33
    program = Program(
        "cooperative_signed_reverse",
        (Cell("source", signed, size), Cell("output", signed, size)),
        (Clock("tick"),),
        (
            Region(
                "map",
                tuple(
                    Store(
                        "output",
                        op(
                            "add",
                            read("source", signed, const(i)),
                            const(100, signed),
                            policy=policy,
                        ),
                        const(size - i - 1),
                    )
                    for i in range(size)
                ),
            ),
        ),
    )
    scenarios = tuple(
        Scenario(
            initial={"source": [(i * 17 + lane * 23) % 256 - 128 for i in range(size)]}
        )
        for lane in range(3)
    )
    expected = session_for(program, scenarios).run(until="1", max_events=4)
    session = session_for(program, scenarios, device="cuda:0", strategy=strategy)
    assert session.cooperative_regions
    assert session.run(until="1", max_events=4) == expected
    assert any(value < 0 for value in expected[0].state["output"])


@CUDA
@pytest.mark.parametrize("strategy", ("warp", "block"))
@pytest.mark.parametrize("poll_interval", (1, 3))
def test_cooperative_compaction_keeps_original_stride_and_terminal_diagnostics(
    strategy, poll_interval
):
    program = map_program("stencil")
    program = replace(
        program,
        inputs=(Input("halt", BOOL), Input("fail", BOOL)),
        stop=inp("halt", BOOL),
        schedule=(
            program.schedule[0],
            Region("check", (Require(op("not", inp("fail", BOOL)), "lane failure"),)),
            *program.schedule[1:],
        ),
    )
    schedules = (
        (Drive("0", {"halt": 1}),),
        (Drive("1/2", {"fail": 1}),),
        (Drive("2/3", {"halt": 1}),),
        (),
    )
    scenarios = tuple(
        replace(s, drives=schedules[i % len(schedules)])
        for i, s in enumerate(scenarios_for(33, 37))
    )
    expected = session_for(program, scenarios).run(until="2", max_events=20)
    session = session_for(program, scenarios, device="cuda:0", strategy=strategy)
    assert session.cooperative_regions
    assert (
        session.campaign(
            until="2",
            max_events=20,
            quantum=1,
            compact=True,
            poll_interval=poll_interval,
        )
        == expected
    )
    assert [r.status for r in expected[:4]] == [
        "complete",
        "fault",
        "complete",
        "horizon",
    ]


@CUDA
@pytest.mark.parametrize(
    "failure", (None, "requirement", "earlier_alias", "later_alias")
)
def test_one_thread_recovered_map_preserves_candidates_and_fault_order(failure):
    program = map_program("stencil")
    map_region = program.schedule[1]
    if failure == "requirement":
        body = map_region.body + (Require(const(0, BOOL), "late map failure"),)
    elif failure == "earlier_alias":
        body = (Store("output", const(99, WORD), inp("alias", WORD)),) + map_region.body
    elif failure == "later_alias":
        body = map_region.body + (Store("output", const(99, WORD), const(7)),)
    else:
        body = map_region.body
    program = replace(
        program,
        inputs=(Input("alias", WORD, 7),),
        schedule=(
            program.schedule[0],
            replace(map_region, body=body),
            program.schedule[2],
        ),
    )
    scenarios = scenarios_for(33, 3)
    expected = session_for(program, scenarios, trace_mode="full").run(
        until="1", max_events=8
    )
    session = Session(
        program,
        scenarios,
        device="cuda:0",
        strategy="block",
        block_size=1,
        trace_mode="full",
        mode=wp.config.mode,
    )
    assert session.cooperative_regions
    assert session.run(until="1", max_events=8) == expected
    if failure:
        assert all(
            r.status == "fault" and r.state["output"] == (0,) * 33 for r in expected
        )
        assert all(r.state["prefix"] == (1,) for r in expected)
    else:
        assert all(
            r.status == "horizon" and r.state["prefix"] == (4,) for r in expected
        )


@CUDA
@pytest.mark.parametrize("block_size", (1, 32, 128))
def test_recovered_map_preserves_zero_delay_microsteps(block_size):
    program = map_program("stencil")
    program = replace(
        program,
        clocks=(),
        timers=(Timer("pulse", "0"),),
        max_microstep=2,
        schedule=(
            *program.schedule,
            Region(
                "again",
                (Schedule("pulse", duration("0"), op("add", microstep(), const(1))),),
            ),
        ),
    )
    scenarios = scenarios_for(33, 3)
    oracle = session_for(program, scenarios)
    session = Session(
        program,
        scenarios,
        device="cuda:0",
        strategy="block",
        block_size=block_size,
        trace_mode="summary",
        mode=wp.config.mode,
    )
    assert session.cooperative_regions
    for _ in range(4):
        session.advance(until="1", max_events=1)
        expected = oracle.step(until="1", max_events=1)
    assert session.inspect() == expected
    assert all(
        r.microstep == 2 and r.fault.fault == "microstep_limit" for r in expected
    )
    assert all(r.state["prefix"] == (3,) for r in expected)
