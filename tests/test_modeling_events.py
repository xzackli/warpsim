"""Independent composed-timer and repeated-transaction contract checks."""

from dataclasses import replace
from fractions import Fraction

import pytest
import warp as wp

from warpsim.core import (
    BOOL,
    Bits,
    Cell,
    Clock,
    Timer,
    Region,
    Repeat,
    Schedule,
    Cancel,
    Store,
    Require,
    Observe,
    const,
    read,
    inp,
    due,
    duration,
    generation,
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


DEVICES = ["reference", "cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
BYTE = Bits(8)


def execute(elaboration, device, **options):
    expected = run_reference(elaboration.program, **options)
    if device != "reference":
        assert run_native(elaboration.program, device=device, mode=wp.config.mode, **options) == expected
    return expected


def counter_component():
    return Component(
        "timed_counter",
        (Cell("count", BYTE),),
        ("clock",),
        (Port("sampled", BYTE, "clock"),),
        (
            Output("value", read("count", BYTE), "clock"),
            Output("active", op("and", due("wake"), op("gt", read("count", BYTE), const(0, BYTE))), "clock"),
        ),
        (
            Region("load", (Store("count", const(3, BYTE)), Schedule("wake", duration("1/3"))), due("clock")),
            Region("drain", (
                Store("count", op("sub", read("count", BYTE), const(1, BYTE))),
                Observe("drained", (read("count", BYTE), inp("sampled", BYTE), generation("wake"), generation("wake", captured=True))),
            ), due("wake")),
        ),
        timers=(Timer("wake"),),
    )


def counter_model(*, pair=False, component=None, limit=3):
    component = counter_component() if component is None else component
    names = ("left", "right") if pair else ("left",)
    transactions = (
        *(Transaction(f"load_{name}", (ActionRef(name, "load"),)) for name in names),
        *(Transaction(f"drain_{name}", (ActionRef(name, "drain"),), PortRef(name, "active"), limit) for name in names),
    )
    return Model(
        "timed_counters",
        tuple(Clock(f"tick_{name}", "2", "0" if name == "left" else "1/6") for name in names),
        (),
        tuple(Instance(name, component, (("clock", f"tick_{name}"),)) for name in names),
        tuple(Binding(PortRef(name, "sampled"), PortRef(name, "value")) for name in names),
        transactions,
        tuple((a.name, b.name) for a, b in zip(transactions, transactions[1:])),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_scoped_timers_formal_clocks_and_repeated_current_outputs(device):
    model = counter_model(pair=True)
    elaboration = elaborate(model)
    assert verify_schedule(elaboration.program, elaboration.certificate)
    assert verify_elaboration(model, elaboration)
    assert tuple(t.name for t in elaboration.program.timers) == ("left__wake", "right__wake")
    result = execute(elaboration, device, until="1/2")
    assert result.status == "horizon" and result.events == 4
    assert result.state == {"left__count": (0,), "right__count": (0,)}
    assert result.calendar["timers"] == {"left__wake": (None, 1), "right__wake": (None, 1)}
    observed = [r for r in result.records if r.kind == "observe"]
    assert [(r.time, r.observation, r.values, r.iteration) for r in observed] == [
        (time, f"{name}__drained", (old, 3, 1, 1), iteration)
        for name, time in (("left", Fraction(1, 3)), ("right", Fraction(1, 2)))
        for iteration, old in enumerate((3, 2, 1))
    ]
    for record in observed:
        origin = elaboration.origin(record.location)
        assert origin.instance == record.observation.split("__")[0]
        assert origin.component == "timed_counter" and origin.action == "drain"
        assert origin.source_path == "body/1"


def failing_pair(*, together):
    first = replace(counter_component(), inputs=(), outputs=(), timers=(Timer("wake", "5"),),
                    actions=(Region("act", (Store("count", const(3, BYTE)), Schedule("wake", duration("1/3")), Observe("published")), due("clock")),))
    second = replace(first, name="failing_counter", actions=(
        Region("act", (Store("count", const(2, BYTE)), Cancel("wake"), Observe("discarded"), Require(const(0, BOOL), "later participant")), due("clock")),
    ))
    refs = (ActionRef("left", "act"), ActionRef("right", "act"))
    transactions = ((Transaction("combined", refs),) if together else
                    (Transaction("first", refs[:1]), Transaction("second", refs[1:])))
    return Model("timer_atomicity", (Clock("tick", "2"),), (),
                 (Instance("left", first, (("clock", "tick"),)), Instance("right", second, (("clock", "tick"),))),
                 (), transactions, () if together else (("first", "second"),))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("together", (True, False))
def test_timer_state_and_observations_share_atomic_domain_and_prefix(device, together):
    model = failing_pair(together=together)
    elaboration = elaborate(model)
    result = execute(elaboration, device, until="0")
    assert result.status == "fault" and result.fault.message == "later participant"
    assert result.state == {"left__count": ((0,) if together else (3,)), "right__count": (0,)}
    assert result.calendar["timers"]["left__wake"] == (((15, 0), 0) if together else ((1, 0), 1))
    assert result.calendar["timers"]["right__wake"] == ((15, 0), 0)
    assert [r.observation for r in result.records if r.kind == "observe"] == ([] if together else ["left__published"])
    origin = elaboration.origin(result.fault.location)
    assert (origin.instance, origin.action, origin.source_path) == ("right", "act", "body/3")
    assert verify_elaboration(model, elaboration)


@pytest.mark.parametrize("device", DEVICES)
def test_repeated_failure_keeps_committed_rounds_and_timer_generation(device):
    component = counter_component()
    drain = component.actions[1]
    component = replace(component, actions=(component.actions[0], replace(drain, body=(
        drain.body[0], Schedule("wake", duration("1/3")), drain.body[1],
        Require(op("gt", read("count", BYTE), const(1, BYTE)), "last round fails"),
    ))))
    model = counter_model(component=component)
    elaboration = elaborate(model)
    result = execute(elaboration, device, until="1")
    assert result.state == {"left__count": (1,)}
    assert result.fault.iteration == 2 and result.fault.time == Fraction(1, 3)
    assert result.calendar["timers"]["left__wake"] == ((2, 0), 3)
    assert [r.values for r in result.records if r.kind == "observe"] == [(3, 3, 1, 1), (2, 3, 2, 1)]
    origin = elaboration.origin(result.fault.location)
    assert (origin.instance, origin.action, origin.source_path) == ("left", "drain", "body/3")
    assert verify_schedule(elaboration.program, elaboration.certificate)


@pytest.mark.parametrize("device", DEVICES)
def test_repeat_bound_fault_has_transaction_origin(device):
    model = counter_model(limit=2)
    elaboration = elaborate(model)
    result = execute(elaboration, device, until="1")
    assert result.fault.fault == "loop_limit" and result.fault.iteration == 2
    assert result.state == {"left__count": (1,)}
    origin = elaboration.origin(result.fault.location)
    assert origin.action == "drain_left" and origin.source_path == "transactions/drain_left"


def test_repeated_commit_records_have_transaction_provenance():
    elaboration = elaborate(counter_model())
    result = run_reference(elaboration.program, until="1/3")
    commits = [r for r in result.records if r.kind == "commit" and r.iteration is not None]
    assert len(commits) == 3
    for record in commits:
        origin = elaboration.origin(record.location)
        assert origin is not None
        assert origin.action == "drain_left" and origin.source_path == "transactions/drain_left"


@pytest.mark.parametrize("device", DEVICES)
def test_repeated_guard_fault_has_transaction_provenance(device):
    component = counter_component()
    bad_guard = op("eq", read("count", BYTE, const(1)), const(0, BYTE))
    component = replace(component, actions=(component.actions[0], replace(component.actions[1], guard=bad_guard)))
    elaboration = elaborate(counter_model(component=component))
    result = execute(elaboration, device, until="1/3")
    assert result.fault.fault == "bounds" and result.fault.iteration == 0
    assert result.state["left__count"] == (3,)
    origin = elaboration.origin(result.fault.location)
    assert origin is not None
    assert origin.action == "drain_left" and origin.source_path == "transactions/drain_left"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("condition, expected_count, fault", ((None, 1, None), (const(0, BOOL), 0, None), (const(1, BOOL), 2, "loop_limit")))
def test_repeat_accepts_closed_expression_or_none(device, condition, expected_count, fault):
    component = Component("increment", (Cell("count", BYTE),), ("clock",), (), (), (
        Region("act", (Store("count", op("add", read("count", BYTE), const(1, BYTE))),)),
    ))
    model = Model("repeat_expressions", (Clock("tick"),), (),
                  (Instance("counter", component, (("clock", "tick"),)),), (),
                  (Transaction("work", (ActionRef("counter", "act"),), condition, 2),))
    elaboration = elaborate(model)
    assert isinstance(elaboration.program.schedule[0], Repeat) == (condition is not None)
    result = execute(elaboration, device, until="0")
    assert result.state["counter__count"] == (expected_count,)
    assert (result.fault.fault if result.fault else None) == fault


@pytest.mark.parametrize("effect", (Schedule("right__wake", duration("1")), Cancel("right__wake"), Observe("peek", (due("right__wake"),)), Observe("peek", (generation("right__wake"),))))
def test_component_cannot_access_another_instances_timer(effect):
    component = counter_component()
    illegal = replace(component, actions=(replace(component.actions[0], body=(effect,)), component.actions[1]))
    with pytest.raises(ValueError):
        elaborate(counter_model(component=illegal))


@pytest.mark.parametrize("timers", ((Timer("wake"), Timer("wake")), (Timer("count"),), (Timer("clock"),)))
def test_timer_namespace_collisions_are_rejected(timers):
    with pytest.raises(ValueError, match="Duplicate symbol"):
        elaborate(counter_model(component=replace(counter_component(), timers=timers)))


@pytest.mark.parametrize("device", DEVICES)
def test_combined_methods_cannot_publish_two_effects_to_one_timer(device):
    component = Component("timer_conflict", (Cell("count", BYTE),), ("clock",), (), (), (
        Region("arm", (Store("count", const(1, BYTE)), Schedule("wake", duration("1/3")), Observe("discarded")), due("clock")),
        Region("clear", (Cancel("wake"),), due("clock")),
    ), timers=(Timer("wake", "5"),))
    model = Model("timer_conflict", (Clock("tick"),), (),
                  (Instance("counter", component, (("clock", "tick"),)),), (),
                  (Transaction("combined", (ActionRef("counter", "arm"), ActionRef("counter", "clear"))),))
    elaboration = elaborate(model)
    result = execute(elaboration, device, until="0")
    assert result.fault.fault == "conflict" and result.fault.target == "counter__wake"
    assert result.state == {"counter__count": (0,)}
    assert result.calendar["timers"]["counter__wake"] == ((15, 0), 0)
    assert not any(r.kind == "observe" for r in result.records)
    origin = elaboration.origin(result.fault.location)
    assert origin.action == "clear" and origin.source_path == "body/0"


@pytest.mark.parametrize("limit", (0, 1025, True, "3"))
def test_repeat_limits_require_bounded_integers(limit):
    with pytest.raises(ValueError):
        elaborate(counter_model(limit=limit))


def test_repeat_and_timer_changes_invalidate_certificates():
    model = counter_model()
    elaboration = elaborate(model)
    repeat = elaboration.program.schedule[1]
    with pytest.raises(ValueError, match="certificate"):
        verify_schedule(replace(elaboration.program, schedule=(elaboration.program.schedule[0], replace(repeat, limit=2))), elaboration.certificate)
    with pytest.raises(ValueError, match="certificate"):
        verify_schedule(replace(elaboration.program, timers=(replace(elaboration.program.timers[0], initial="2"),)), elaboration.certificate)
    with pytest.raises(ValueError, match="Elaboration"):
        verify_elaboration(model, replace(elaboration, origins=elaboration.origins[:-1]))
