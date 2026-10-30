"""Regressions from independent execution and measurement review."""

import importlib.util
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import warp as wp
from warpsim.core import (
    Bits,
    Cell,
    Clock,
    Drive,
    Program,
    Region,
    Scenario,
    Session,
    Store,
    const,
    op,
    read,
)
from warpsim.campaign_fixtures import build_case


def test_reference_result_does_not_mutate_engine():
    word = Bits(8)
    p = Program(
        "owning",
        (Cell("x", word),),
        (Clock("tick"),),
        (
            Region(
                "increment", (Store("x", op("add", read("x", word), const(1, word))),)
            ),
        ),
    )
    s = Session(p, device="reference", trace_mode="summary")
    first = s.step(until="0")[0]
    first.state["x"] = (40,)
    assert s.step(until="1")[0].state["x"] == (2,)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is required")
@pytest.mark.parametrize("strategy", ("scalar", "warp", "block"))
def test_session_and_restore_keep_stream_ownership_when_callers_switch(
    strategy, tmp_path
):
    case = build_case("map", batch=37, size=33, events=12)
    oracle = Session(
        case.program,
        case.scenarios,
        device="reference",
        trace_mode="summary",
        trace_cells=case.trace_cells,
    )
    owner, caller, restore_owner = (wp.Stream("cuda:0") for _ in range(3))
    with wp.ScopedStream(owner, sync_enter=False, sync_exit=False):
        session = Session(
            case.program,
            case.scenarios,
            device="cuda:0",
            strategy=strategy,
            mode=wp.config.mode,
            trace_mode="summary",
            trace_cells=case.trace_cells,
        )
    assert session.stream is owner
    with wp.ScopedStream(caller, sync_enter=False, sync_exit=False):
        session.advance(until=case.until, max_events=3)
        expected = oracle.step(until=case.until, max_events=3)
        assert session.status()[0].events == expected[0].events
        path = tmp_path / "stream.json"
        session.checkpoint(path)
        assert session.inspect() == expected
    with wp.ScopedStream(restore_owner, sync_enter=False, sync_exit=False):
        restored = Session.restore(
            path, device="cuda:0", strategy=strategy, mode=wp.config.mode
        )
    assert restored.stream is restore_owner
    with wp.ScopedStream(caller, sync_enter=False, sync_exit=False):
        actual = restored.campaign(
            until=case.until, max_events=9, quantum=2, compact=True
        )
        assert actual == oracle.run(until=case.until, max_events=9)
        assert session.inspect() == expected


def test_restore_reuses_allocated_state(tmp_path, monkeypatch):
    p = Program("bounded_restore", (Cell("x", Bits(8), 256),), (), ())
    s = Session(
        p, device="reference", trace_mode="summary", trace_cells=(), memory_limit=2400
    )
    s.run(until="0")
    path = tmp_path / "state.json"
    s.checkpoint(path)
    original = Session._initialize_native
    pointers = []

    def capture(self):
        original(self)
        pointers.append(self.states.ptr)

    monkeypatch.setattr(Session, "_initialize_native", capture)
    restored = Session.restore(path, device="cpu", memory_limit=2400)
    assert restored.states.ptr == pointers[0]
    assert restored.metrics["peak_buffer_bytes"] <= 2400
    assert restored.inspect() == s.inspect()


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_count_only_compaction_matches_indexed(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    case = build_case("skew", batch=37, events=20)
    outputs = []
    counts = []
    for filtered in (False, True):
        s = Session(
            case.program,
            case.scenarios,
            device=device,
            trace_mode="summary",
            trace_cells=case.trace_cells,
            mode=wp.config.mode,
        )
        outputs.append(
            s.campaign(
                until=case.until,
                max_events=case.max_events,
                quantum=2,
                compact=True,
                filter_active=filtered,
            )
        )
        counts.append(s.active_counts)
    assert (
        outputs[0] == outputs[1] == case.run(device="reference", trace_mode="summary")
    )
    assert counts[0] == counts[1] and len(counts[0]) > 1


def test_benchmark_oracle_shares_batch_lattice_and_counts_budget_results():
    spec = importlib.util.spec_from_file_location(
        "benchmark_review", Path(__file__).parents[1] / "scripts/benchmark_core.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    case = build_case("control", batch=2, events=20)
    scenarios = tuple(
        replace(
            s,
            drives=tuple(
                sorted(s.drives + (Drive(t, {}),), key=lambda d: Fraction(d.time))
            ),
        )
        for s, t in zip(case.scenarios, ("1/2", "1/3"))
    )
    case = replace(case, scenarios=scenarios, max_events=2)
    expected, work = benchmark.reference_work(case)
    assert expected == case.run(device="reference", trace_mode="summary")
    samples = [dict.fromkeys(benchmark.TIMING_KEYS, 1.0)] * 3
    summary = benchmark.summarize(samples, work)
    assert summary["outcomes_per_campaign_result_second"]["median"] == 2
    assert (
        summary["outcomes_by_status_per_campaign_result_second"]["budget"]["median"]
        == 2
    )


@pytest.mark.parametrize("value", ("not an integer", True, 256, -1))
def test_checkpoint_diagnostic_operands_follow_declared_type(tmp_path, value):
    from warpsim.core import BOOL, Require
    from warpsim.core.serialization import load, save

    p = Program(
        "diagnostic_type",
        (Cell("x", Bits(8)),),
        (Clock("tick"),),
        (Region("fail", (Require(const(0, BOOL), "failed", (read("x", Bits(8)),)),)),),
    )
    s = Session(p, device="reference", trace_mode="summary")
    s.run(until="0")
    path = tmp_path / "state.json"
    s.checkpoint(path)
    data = load(path, "warpsim.checkpoint/v2")
    result = data["results"][0]
    data["results"] = (
        replace(result, records=(replace(result.records[0], values=(value,)),)),
    )
    save(path, "warpsim.checkpoint/v2", data)
    with pytest.raises(ValueError, match="Expected integer"):
        Session.restore(path, device="cpu")
