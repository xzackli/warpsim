"""Semantic boundary cases with independent expected results on CPU and CUDA."""

from collections import deque
from dataclasses import replace
from fractions import Fraction
import random

import pytest
import warp as wp

from warpsim.core import (
    Bits,
    BOOL,
    TIME,
    Cell,
    Input,
    Clock,
    Program,
    Region,
    Repeat,
    Let,
    Store,
    Require,
    Observe,
    If,
    Drive,
    const,
    read,
    inp,
    local,
    due,
    op,
    lower,
    run_reference,
    run_native,
)
from warpsim.core.codegen import generate
from warpsim.models.queued_fir import queued_fir

U = Bits(8)
DEVICES = ["reference", "cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


@pytest.fixture(params=DEVICES)
def execute(request):
    def run(p, **kwargs):
        expected = run_reference(p, **kwargs)
        if request.param == "reference":
            return expected
        actual = run_native(p, device=request.param, mode=wp.config.mode, **kwargs)
        assert actual == expected
        return actual

    return run


def program(schedule, *, cells=(Cell("x", U), Cell("y", U)), captures=(), inputs=()):
    return Program("litmus", cells, (Clock("clock"),), schedule, inputs, captures)


def observed(result):
    return [
        (r.time, r.observation, r.values) for r in result.records if r.kind == "observe"
    ]


def test_entry_reads_captures_and_sequential_commits(execute):
    p = program(
        (
            Region("first", (Store("x", const(9, U)), Store("y", read("x", U)))),
            Region(
                "second",
                (Observe("values", (read("x", U), read("y", U), local("old", U))),),
            ),
        ),
        captures=(Let("old", read("x", U)),),
    )
    result = execute(p, until="0", initial={"x": 4})
    assert result.state == {"x": (9,), "y": (4,)}
    assert observed(result) == [(0, "values", (9, 4, 4))]


def test_failed_region_discards_writes_and_observations_keeps_prefix(execute):
    p = program(
        (
            Region("first", (Store("x", const(9, U)), Observe("success"))),
            Region(
                "second",
                (
                    Store("y", const(7, U)),
                    Observe("unpublished"),
                    Require(const(0, BOOL), "deliberate failure"),
                ),
            ),
        )
    )
    result = execute(p, until="0")
    assert result.status == "fault"
    assert result.state == {"x": (9,), "y": (0,)}
    assert observed(result) == [(0, "success", ())]
    assert result.fault.region == "second"
    assert result.fault.location == "schedule/1/2"
    # This semantic mutation must produce a different committed prefix.
    reordered = execute(replace(p, schedule=p.schedule[::-1]), until="0")
    assert reordered.state["x"] == (0,)


@pytest.mark.parametrize("alias", [False, True])
def test_indexed_alias_conflict_and_old_load(execute, alias):
    index = const(0 if alias else 1)
    p = program(
        (
            Region(
                "writes",
                (
                    Store("m", const(4, U), const(0)),
                    Store("m", const(4, U), index),
                    Store("y", read("m", U, const(0))),
                ),
            ),
        ),
        cells=(Cell("m", U, 2, (8, 9)), Cell("y", U)),
    )
    r = execute(p, until="0")
    if alias:
        assert r.status == "fault" and r.fault.fault == "conflict"
        assert r.fault.target == "m" and r.fault.index == 0
        assert r.state == {"m": (8, 9), "y": (0,)}
    else:
        assert r.state == {"m": (4, 4), "y": (8,)}


def test_disabled_and_unselected_invalid_reads_are_not_evaluated(execute):
    invalid = read("m", U, const(2))
    p = program(
        (
            Region("disabled", (Store("y", invalid),), const(0, BOOL)),
            Region(
                "active",
                (
                    Store("y", op("select", const(0, BOOL), invalid, const(5, U))),
                    If(const(0, BOOL), (Store("m", const(1, U), const(3)),)),
                ),
            ),
        ),
        cells=(Cell("m", U, 2), Cell("y", U)),
    )
    r = execute(p, until="0")
    assert r.status == "horizon" and r.state["y"] == (5,)
    assert r.records[0].kind == "disabled"


def test_first_fault_priority_and_atomicity(execute):
    p = program(
        (
            Region(
                "bad",
                (
                    Store("x", const(3, U)),
                    Store("y", op("add", read("x", U), read("m", U, const(2)))),
                ),
            ),
        ),
        cells=(Cell("x", U, initial=(255,)), Cell("y", U), Cell("m", U)),
    )
    r = execute(p, until="0")
    assert r.fault.fault == "bounds" and r.fault.target == "m"
    assert r.state["x"] == (255,) and r.state["y"] == (0,)


def test_repeated_exp_style_rounds_keep_successful_prefix(execute):
    p = program(
        (
            Repeat(
                "drops",
                op("ge", read("x", U), const(4, U)),
                8,
                (
                    Region(
                        "round",
                        (
                            Require(
                                op("ge", read("y", U), const(4, U)),
                                "all channels must have four",
                            ),
                            Store("x", op("sub", read("x", U), const(4, U))),
                            Store("y", op("sub", read("y", U), const(4, U))),
                            Observe("dropped", (read("x", U), read("y", U))),
                        ),
                    ),
                ),
            ),
        )
    )
    r = execute(p, until="0", initial={"x": 12, "y": 7})
    assert r.state == {"x": (8,), "y": (3,)}
    assert r.fault.fault == "requirement"
    assert observed(r) == [(0, "dropped", (12, 7))]


def test_nonprogress_bound_is_fault_not_completion(execute):
    p = program(
        (
            Repeat(
                "stuck",
                const(1, BOOL),
                3,
                (Region("attempt", (Store("x", const(1, U)),)),),
            ),
        )
    )
    r = execute(p, until="0")
    assert r.status == "fault" and r.fault.fault == "loop_limit"
    assert len([x for x in r.records if x.kind == "commit"]) == 3
    assert r.state["x"] == (1,)


@pytest.mark.parametrize("completion_first", [True, False])
def test_ordered_completion_read_collision(execute, completion_first):
    complete = Region("complete", (Store("x", op("add", read("x", U), const(4, U))),))
    consume = Region(
        "consume",
        (
            Require(op("gt", read("x", U), const(0, U)), "underflow"),
            Store("x", op("sub", read("x", U), const(1, U))),
            Observe("read"),
        ),
    )
    r = execute(
        program((complete, consume) if completion_first else (consume, complete)),
        until="0",
    )
    assert r.status == ("horizon" if completion_first else "fault")
    assert r.state["x"] == ((3,) if completion_first else (0,))


def test_exact_calendar_gathers_due_set_and_samples_input_first(execute):
    p = Program(
        "calendar",
        (Cell("x", U),),
        (Clock("a", "1/3"), Clock("b", "1/2")),
        (Region("record", (Observe("due", (due("a"), due("b"), inp("data", U))),)),),
        (Input("data", U),),
    )
    r = execute(
        p, until="1", drives=(Drive("1/6", {"data": 5}), Drive("1", {"data": 9}))
    )
    assert observed(r) == [
        (0, "due", (1, 1, 0)),
        (Fraction(1, 6), "due", (0, 0, 5)),
        (Fraction(1, 3), "due", (1, 0, 5)),
        (Fraction(1, 2), "due", (0, 1, 5)),
        (Fraction(2, 3), "due", (1, 0, 5)),
        (1, "due", (1, 1, 9)),
    ]
    # Declaration order has no influence on coincidence collection.
    other = execute(
        replace(p, clocks=p.clocks[::-1]),
        until="1",
        drives=(Drive("1/6", {"data": 5}), Drive("1", {"data": 9})),
    )
    assert other.records == r.records


def test_budget_horizon_empty_calendar_and_large_time(execute):
    p = program((Region("tick", (Observe("tick"),)),))
    r = execute(p, until="2", max_events=2)
    assert r.status == "budget" and r.time == 1 and r.events == 2
    assert execute(p, until="1", max_events=2).status == "horizon"
    assert execute(replace(p, clocks=()), until="1").events == 0
    maximum = (1 << 62) - 1
    large = replace(p, clocks=(Clock("clock", str(maximum), str(maximum)),))
    r = execute(large, until=str(maximum))
    assert r.status == "horizon" and r.time == maximum and r.events == 1


def test_signed_rounding_wrap_saturation_over_full_small_domain(execute):
    small, narrow = Bits(5, True), Bits(3, True)
    x = inp("data", small)
    p = program(
        (
            Region(
                "arithmetic",
                (
                    Observe(
                        "values",
                        (
                            op("round_even", x, const(1)),
                            op("cast", x, type=narrow, policy="wrap"),
                            op("cast", x, type=narrow, policy="saturate"),
                        ),
                    ),
                ),
            ),
        ),
        inputs=(Input("data", small),),
    )
    r = execute(
        p,
        until="31",
        drives=tuple(
            Drive(str(i), {"data": value}) for i, value in enumerate(range(-16, 16))
        ),
    )
    for (_, _, values), x in zip(observed(r), range(-16, 16)):
        assert values == (round(Fraction(x, 2)), ((x + 4) % 8) - 4, max(-4, min(3, x)))


def test_checked_narrowing_faults_transaction(execute):
    p = program(
        (
            Region(
                "overflow",
                (
                    Store("x", const(7, U)),
                    Store("y", op("cast", const(-1, Bits(3, True)), type=U)),
                ),
            ),
        )
    )
    r = execute(p, until="0")
    assert r.fault.fault == "overflow" and r.state["x"] == (0,)


def test_empty_queue_no_fallthrough_and_separate_repair_rollback(execute):
    p = queued_fir()
    r = execute(
        p,
        until="0",
        drives=(Drive("0", {"valid": 1, "value": 3, "id": 7}),),
        initial={"repair_pending": 1, "repair_address": 2},
    )
    assert r.fault.region == "repair" and r.fault.fault == "bounds"
    assert r.state["q_count"] == (1,) and r.state["out_valid"] == (0,)
    assert r.state["repair_pending"] == (1,)
    assert observed(r) == [(0, "accepted", (7, 3, 0))]


def test_full_ring_exchange_alias_and_losing_invalid_repair(execute):
    p = queued_fir()
    initial = {
        "q_count": 2,
        "q_value": (3, 4),
        "q_id": (10, 11),
        "q_time": (5, 6),
        "history": (5, 0),
        "history_id": (2, 0),
        "repair_pending": 1,
        "repair_address": 2,
        "out_valid": 1,
        "out_value": 99,
        "out_id": 9,
        "out_time": 4,
    }
    r = execute(
        p,
        until="0",
        initial=initial,
        drives=(Drive("0", {"valid": 1, "value": 6, "id": 12, "source_time": 7}),),
    )
    assert r.status == "horizon" and r.state["repair_pending"] == (1,)
    assert r.state["q_count"] == (2,) and r.state["q_value"] == (6, 4)
    assert r.state["history"] == (3, 0) and r.state["out_value"] == (8,)
    assert observed(r) == [
        (0, "consumed", (9, 99, 4, 0)),
        (0, "produced", (10, 8, 5, 2)),
        (0, "accepted", (12, 6, 7)),
    ]


def test_stalled_output_metadata_is_stable(execute):
    initial = {
        "q_count": 1,
        "q_value": (3, 0),
        "out_valid": 1,
        "out_value": 99,
        "out_id": 9,
        "out_time": 4,
        "out_history_id": 8,
    }
    r = execute(
        queued_fir(), until="2", initial=initial, drives=(Drive("0", {"ready": 0}),)
    )
    for name in ("out_valid", "out_value", "out_id", "out_time", "out_history_id"):
        assert r.state[name] == (initial[name],)
    assert r.state["q_count"] == (1,) and not observed(r)


def test_stream_fault_rolls_back_retirement_too(execute):
    r = execute(
        queued_fir(),
        until="0",
        initial={"out_valid": 1, "out_id": 12, "q_count": 1, "q_address": (2, 0)},
    )
    assert r.fault.fault == "bounds" and r.fault.region == "edge"
    assert r.state["out_valid"] == (1,) and r.state["q_count"] == (1,)
    assert not observed(r)


def test_large_indexed_storage_uses_fixed_sparse_journal(execute):
    p = queued_fir(history_size=2048)
    assert lower(p).journal_size == lower(queued_fir()).journal_size
    history = [0] * 2048
    history[-1] = 5
    r = execute(
        p,
        until="0",
        max_events=1,
        initial={
            "q_count": 1,
            "q_value": (3, 0),
            "q_address": (len(history) - 1, 0),
            "history": history,
        },
    )
    assert r.state["out_value"] == (8,) and r.state["history"][-1] == 3
    assert not any(r.state["history"][:-1])


@pytest.mark.parametrize(
    "sample,history,coefficients,expected",
    [
        (3, 5, (1024, 1024), 8),
        (-512, -512, (-2048, -2048), 511),
        (-1, 0, (512, 0), 0),
        (-3, 0, (512, 0), -2),
        (511, 511, (2047, 2047), 511),
    ],
)
def test_fir_signed_edges(execute, sample, history, coefficients, expected):
    r = execute(
        queued_fir(coefficients=coefficients),
        until="0",
        initial={"q_count": 1, "q_value": (sample, 0), "history": (history, 0)},
    )
    assert r.state["out_value"] == (expected,)


@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_queued_fir_against_independent_token_model(execute, capacity):
    # Exact clocks and a separate deque model; no core lowering or arithmetic helpers.
    rng = random.Random(123)
    drives = tuple(
        Drive(
            str(Fraction(k, 6)),
            {
                "valid": int(rng.random() < 0.8),
                "value": rng.randrange(-12, 12),
                "id": k,
                "source_time": k,
                "address": k % 2,
                "ready": int(rng.random() < 0.6),
            },
        )
        for k in range(61)
    )
    q, history, out, expected = deque(), [(0, 0), (0, 0)], None, []
    for k, d in enumerate(drives):
        t, x = Fraction(k, 6), d.values
        retire = k % 4 == 0 and out is not None and x["ready"]
        take = k % 4 == 0 and bool(q) and (out is None or x["ready"])
        put = k % 3 == 0 and x["valid"] and (len(q) < capacity or take)
        if retire:
            expected.append((t, "consumed", out))
        if take:
            value, uid, birth, addr = q.popleft()
            old, old_id = history[addr]
            out = (uid, value + old, birth, old_id)
            history[addr] = (value, uid)
            expected.append((t, "produced", out))
        elif retire:
            out = None
        if put:
            q.append((x["value"], x["id"], x["source_time"], x["address"]))
            expected.append((t, "accepted", (x["id"], x["value"], x["source_time"])))
    r = execute(
        queued_fir(capacity=capacity, producer_period="1/2", consumer_period="2/3"),
        until="10",
        drives=drives,
    )
    assert observed(r) == expected
    assert r.state["q_count"] == (len(q),)
    assert r.state["history"] == tuple(v for v, _ in history)


def test_lowering_rejects_unsafe_or_ambiguous_programs():
    with pytest.raises(ValueError, match="carrier"):
        lower(
            program(
                (
                    Region(
                        "bad",
                        (
                            Store(
                                "x", op("mul", inp("a", TIME), inp("a", TIME), type=U)
                            ),
                        ),
                    ),
                ),
                inputs=(Input("a", TIME),),
            )
        )
    with pytest.raises(ValueError, match="BOOL"):
        lower(program((Region("bad", (), const(1, U)),)))
    with pytest.raises(ValueError, match="Unknown local"):
        lower(
            program(
                (
                    Region("a", (Let("secret", const(1, U)),)),
                    Region("b", (Store("x", local("secret", U)),)),
                )
            )
        )
    with pytest.raises(ValueError, match="type mismatch"):
        lower(program((Region("bad", (Store("x", const(1, BOOL)),)),)))
    with pytest.raises(ValueError, match="Unsupported expression"):
        lower(
            program((Region("bad", (Store("x", op("python_callback", const(1, U))),)),))
        )
    with pytest.raises(ValueError, match="scalar Let"):
        lower(program((), captures=(Store("x", const(1, U)),)))


def test_run_input_validation():
    p = queued_fir()
    for fn in (run_reference, run_native):
        with pytest.raises(ValueError, match="strictly increasing"):
            fn(p, until="1", drives=(Drive("0", {}), Drive("0", {})))
        with pytest.raises(ValueError, match="Unknown input"):
            fn(p, until="1", drives=(Drive("0", {"typo": 1}),))
        with pytest.raises(ValueError, match="integer"):
            fn(p, until="1", drives=(Drive("0", {"value": 512}),))
        with pytest.raises(ValueError, match="initial"):
            fn(p, until="1", initial={"typo": 1})
        with pytest.raises(ValueError, match="rational string"):
            fn(p, until=0.1)


def test_declarations_specialize_shared_backend_without_model_names():
    p = queued_fir()
    other = replace(p, name="another_composition")
    # A semantic name affects identity only, not generated execution.
    assert generate(p).splitlines()[1:] == generate(other).splitlines()[1:]
    l = lower(p)
    assert l.journal_size < l.width
    assert "history" in l.effects["edge"]["reads"]
    assert "history" in l.effects["edge"]["writes"]
    assert l.effects["repair"]["may_fault"]


@pytest.mark.parametrize("mutation", ["early_store", "missing_alias_check"])
def test_deliberately_broken_lowering_is_detected(monkeypatch, mutation):
    from warpsim.core.codegen import Emitter

    original = Emitter.generate
    if mutation == "early_store":
        p = program(
            (
                Region(
                    "fails",
                    (Store("x", const(8, U)), Require(const(0, BOOL), "rollback")),
                ),
            )
        )
        before = "if(!fault) { for(int j=0;j<nw;++j) S(DATA+addresses[j])=values[j];"
        after = "for(int j=0;j<nw;++j) S(DATA+addresses[j])=values[j]; if(!fault) {"
    else:
        p = program(
            (Region("conflict", (Store("x", const(8, U)), Store("x", const(8, U)))),)
        )
        before, after = "if(address==addresses[j])", "if(false)"

    def broken(self):
        source = original(self)
        assert before in source
        return source.replace(before, after)

    monkeypatch.setattr(Emitter, "generate", broken)
    expected = run_reference(p, until="0")
    actual = run_native(p, until="0", device="cpu", mode=wp.config.mode)
    assert actual != expected
    assert expected.state["x"] == (0,)
    assert actual.state["x"] == (8,)
