"""Composition contracts, transactional transfers, and independent deque witness."""

from collections import deque
from dataclasses import replace
import subprocess
import sys

import pytest
import warp as wp

from warpsim.core import (
    Bits,
    BOOL,
    Cell,
    Input,
    Clock,
    Region,
    Store,
    Require,
    Observe,
    Drive,
    const,
    read,
    inp,
    due,
    op,
    run_reference,
    run_native,
)
from warpsim.modeling import (
    Port,
    Output,
    Component,
    Instance,
    PortRef,
    Binding,
    ActionRef,
    Transaction,
    Model,
    elaborate,
    verify_schedule,
    verify_elaboration,
)
from warpsim.models.fifo_pipeline import fifo_pipeline

DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
U8 = Bits(8)


def counter_component():
    return Component(
        "counter",
        (Cell("x", U8),),
        ("clock",),
        (Port("enable", BOOL, "clock"),),
        (Output("value", read("x", U8), "clock"),),
        (
            Region(
                "step",
                (Store("x", op("add", read("x", U8), const(1, U8))),),
                op("and", due("clock"), inp("enable", BOOL)),
            ),
        ),
    )


def pair(*, together=True, failing=False):
    a = counter_component()
    b = replace(
        a,
        name="checked",
        actions=(
            replace(
                a.actions[0],
                body=(
                    *a.actions[0].body,
                    Require(const(0 if failing else 1, BOOL), "second participant"),
                ),
            ),
        ),
    )
    instances = (
        Instance("a", a, (("clock", "tick"),)),
        Instance("b", b, (("clock", "tick"),)),
    )
    refs = (ActionRef("a", "step"), ActionRef("b", "step"))
    transactions = (
        (Transaction("edge", refs),)
        if together
        else (Transaction("first", refs[:1]), Transaction("second", refs[1:]))
    )
    return Model(
        "pair",
        (Clock("tick"),),
        (),
        instances,
        tuple(Binding(PortRef(i.name, "enable"), const(1, BOOL)) for i in instances),
        transactions,
        () if together else (("first", "second"),),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_atomic_participants_and_ordered_failure_prefix(device):
    for together in (True, False):
        e = elaborate(pair(together=together, failing=True))
        actual = run_native(e.program, until="0", device=device, mode=wp.config.mode)
        assert actual == run_reference(e.program, until="0")
        assert actual.state["a__x"] == ((0,) if together else (1,))
        assert actual.state["b__x"] == (0,)
        origin = e.origin(actual.fault.location)
        assert origin.instance == "b" and origin.action == "step"
        assert origin.source_path == "body/1"
        assert verify_schedule(e.program, e.certificate)


def test_order_is_explicit_and_certificate_detects_mutation():
    m = pair(together=False)
    with pytest.raises(ValueError, match="Ambiguous"):
        elaborate(replace(m, before=()))
    with pytest.raises(ValueError, match="Cyclic"):
        elaborate(replace(m, before=(("first", "second"), ("second", "first"))))
    e = elaborate(m)
    assert verify_elaboration(m, e)
    with pytest.raises(ValueError, match="certificate"):
        verify_schedule(
            replace(e.program, schedule=e.program.schedule[::-1]), e.certificate
        )
    with pytest.raises(ValueError, match="certificate"):
        verify_schedule(e.program, replace(e.certificate, effects=()))
    with pytest.raises(ValueError, match="Elaboration"):
        verify_elaboration(m, replace(e, origins=()))
    with pytest.raises(ValueError, match="Elaboration"):
        verify_elaboration(
            m, replace(e, certificate=replace(e.certificate, components=()))
        )


def test_ports_check_types_clocks_ownership_and_cycles():
    m = pair()
    with pytest.raises(ValueError, match="type"):
        elaborate(
            replace(
                m,
                bindings=(
                    Binding(PortRef("a", "enable"), const(1, U8)),
                    *m.bindings[1:],
                ),
            )
        )
    with pytest.raises(ValueError, match="Multiple drivers"):
        elaborate(replace(m, bindings=(*m.bindings, m.bindings[0])))
    with pytest.raises(ValueError, match="Every input"):
        elaborate(replace(m, bindings=m.bindings[1:]))
    with pytest.raises(ValueError, match="private"):
        elaborate(
            replace(
                m,
                bindings=(
                    Binding(PortRef("a", "enable"), read("b__x", BOOL)),
                    *m.bindings[1:],
                ),
            )
        )
    illegal = replace(
        m.instances[0].component,
        actions=(Region("step", (Store("b__x", const(1, U8)),)),),
    )
    with pytest.raises(ValueError):
        elaborate(
            replace(
                m,
                instances=(replace(m.instances[0], component=illegal), m.instances[1]),
            )
        )
    relay = Component(
        "relay",
        (Cell("x", BOOL),),
        ("clock",),
        (Port("in", BOOL, "clock"),),
        (Output("out", inp("in", BOOL), "clock"),),
        (Region("step", ()),),
    )
    i = tuple(Instance(n, relay, (("clock", "tick"),)) for n in ("a", "b"))
    cyclic = replace(
        m,
        instances=i,
        bindings=(
            Binding(PortRef("a", "in"), PortRef("b", "out")),
            Binding(PortRef("b", "in"), PortRef("a", "out")),
        ),
    )
    with pytest.raises(ValueError, match="cycle"):
        elaborate(cyclic)
    crossing = replace(
        cyclic,
        clocks=(Clock("tick"), Clock("other", "2")),
        instances=(i[0], replace(i[1], clocks=(("clock", "other"),))),
    )
    with pytest.raises(ValueError, match="clocks differ"):
        elaborate(crossing)


@pytest.mark.parametrize("device", DEVICES)
def test_entry_ports_remain_sampled_across_successive_commits(device):
    writer = counter_component()
    reader = Component(
        "reader",
        (Cell("seen", U8),),
        ("clock",),
        (Port("in", U8, "clock"),),
        (),
        (Region("step", (Store("seen", inp("in", U8)),)),),
    )
    m = Model(
        "sampled",
        (Clock("tick"),),
        (),
        (
            Instance("a", writer, (("clock", "tick"),)),
            Instance("b", reader, (("clock", "tick"),)),
        ),
        (
            Binding(PortRef("a", "enable"), const(1, BOOL)),
            Binding(PortRef("b", "in"), PortRef("a", "value")),
        ),
        (
            Transaction("write", (ActionRef("a", "step"),)),
            Transaction("read", (ActionRef("b", "step"),)),
        ),
        (("write", "read"),),
    )
    e = elaborate(m)
    actual = run_native(e.program, until="3", device=device, mode=wp.config.mode)
    assert actual == run_reference(e.program, until="3")
    assert actual.state["a__x"] == (4,) and actual.state["b__seen"] == (3,)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("capacities", [(1, 1), (2, 3, 1)])
def test_fifo_pipeline_against_independent_deques(device, capacities):
    e = fifo_pipeline(capacities)
    queues = [deque() for _ in capacities]
    drives, expected = [], []
    for tick in range(50):
        valid, sink_ready = tick < 30 and tick % 5 != 2, tick % 7 in (0, 1, 5)
        token = (tick, tick - 17, tick * 2)
        drives.append(
            Drive(
                str(tick),
                dict(
                    valid=int(valid),
                    ready=int(sink_ready),
                    id=token[0],
                    value=token[1],
                    source_time=token[2],
                ),
            )
        )
        old = [tuple(q) for q in queues]
        take, ready = [False] * len(queues), [False] * len(queues)
        for i in reversed(range(len(queues))):
            take[i] = bool(old[i]) and (
                sink_ready if i == len(queues) - 1 else ready[i + 1]
            )
            ready[i] = len(old[i]) < capacities[i] or take[i]
        put = [
            valid and ready[0],
            *(bool(old[i - 1]) and ready[i] for i in range(1, len(queues))),
        ]
        for i, q in enumerate(queues):
            if take[i]:
                expected.append((tick, f"stage{i}__dequeued", q.popleft()))
            if put[i]:
                value = token if i == 0 else old[i - 1][0]
                q.append(value)
                expected.append((tick, f"stage{i}__enqueued", value))
    actual = run_native(
        e.program,
        until="49",
        drives=drives,
        max_events=50,
        device=device,
        mode=wp.config.mode,
    )
    assert actual == run_reference(e.program, until="49", drives=drives, max_events=50)
    observations = [
        (int(r.time), r.observation, r.values)
        for r in actual.records
        if r.kind == "observe"
    ]
    assert observations == expected
    for i, q in enumerate(queues):
        state = actual.state
        head, count = (
            state[f"stage{i}__queue_head"][0],
            state[f"stage{i}__queue_count"][0],
        )
        assert count == len(q)
        got = [
            tuple(
                state[f"stage{i}__queue_{f}"][(head + n) % capacities[i]]
                for f in ("id", "value", "time")
            )
            for n in range(count)
        ]
        assert got == list(q)
    assert verify_schedule(e.program, e.certificate)


@pytest.mark.parametrize("device", DEVICES)
def test_transfer_fault_rolls_back_all_component_owners(device):
    e = fifo_pipeline((1, 1))
    actual = run_native(
        e.program,
        until="0",
        max_events=1,
        device=device,
        mode=wp.config.mode,
        initial={
            "stage0__queue_count": 1,
            "stage0__queue_id": (99,),
            "stage1__queue_head": 1,
        },
    )
    assert actual.status == "fault"
    assert actual.state["stage0__queue_count"] == (1,)
    assert not any(r.kind == "observe" for r in actual.records)


@pytest.mark.parametrize("device", DEVICES)
def test_disabled_participant_does_not_evaluate_later_guard(device):
    m = pair()
    b = m.instances[1]
    bad = op("eq", read("x", U8, const(1)), const(0, U8))
    b = replace(
        b,
        component=replace(
            b.component, actions=(replace(b.component.actions[0], guard=bad),)
        ),
    )
    m = replace(
        m,
        instances=(m.instances[0], b),
        bindings=(replace(m.bindings[0], source=const(0, BOOL)), m.bindings[1]),
    )
    e = elaborate(m)
    actual = run_native(e.program, until="0", device=device, mode=wp.config.mode)
    assert actual == run_reference(e.program, until="0")
    assert actual.fault is None and actual.records[0].kind == "disabled"
    assert actual.state["a__x"] == actual.state["b__x"] == (0,)
    assert e.origin(actual.records[0].location).source_path == "transactions/edge"


@pytest.mark.parametrize("device", DEVICES)
def test_conflicting_methods_on_one_owner_reject_atomic_patch(device):
    c = counter_component()
    c = replace(c, actions=(*c.actions, replace(c.actions[0], name="also_step")))
    m = Model(
        "conflict",
        (Clock("tick"),),
        (),
        (Instance("a", c, (("clock", "tick"),)),),
        (Binding(PortRef("a", "enable"), const(1, BOOL)),),
        (Transaction("edge", (ActionRef("a", "step"), ActionRef("a", "also_step"))),),
    )
    e = elaborate(m)
    actual = run_native(e.program, until="0", device=device, mode=wp.config.mode)
    assert actual == run_reference(e.program, until="0")
    assert actual.fault.fault == "conflict" and actual.state["a__x"] == (0,)
