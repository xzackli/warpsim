"""Zero owned cells are valid for monitors and combinational components."""

from dataclasses import replace
from fractions import Fraction

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
    Timer,
    const,
    due,
    duration,
    generation,
    inp,
    now,
    op,
    read,
    run_native,
    run_reference,
)
from warpsim.core.serialization import Case
from warpsim.modeling import (
    ActionRef,
    Binding,
    Component,
    Instance,
    Model,
    Output,
    Port,
    PortRef,
    Transaction,
    elaborate,
    verify_elaboration,
    verify_schedule,
)


DEVICES = ["reference", "cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
BYTE = Bits(8)


def observer():
    return Program(
        "stateless_observer",
        (),
        (Clock("clock", "1/3"),),
        (
            Region(
                "watch",
                (
                    Observe("sample", (now(), due("clock"), inp("accepted", BOOL))),
                    Require(inp("accepted", BOOL), "accepted input", (now(),)),
                ),
            ),
        ),
        inputs=(Input("accepted", BOOL, 1),),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trace_mode", ("full", "observations", "summary"))
def test_zero_cell_observer_discards_failed_observation_and_preserves_diagnostic(
    device, trace_mode
):
    options = dict(
        until="1",
        drives=(Drive("1/2", {"accepted": 0}),),
        trace_mode=trace_mode,
    )
    expected = run_reference(observer(), **options)
    actual = (
        expected
        if device == "reference"
        else run_native(observer(), device=device, mode=wp.config.mode, **options)
    )
    assert actual == expected
    assert actual.state == {} and all(record.state == {} for record in actual.records)
    assert (
        actual.status == "fault"
        and actual.time == Fraction(1, 2)
        and actual.events == 3
    )
    assert actual.fault.message == "accepted input" and actual.fault.values == (3,)
    observations = [record for record in actual.records if record.kind == "observe"]
    assert [(record.time, record.values) for record in observations] == (
        []
        if trace_mode == "summary"
        else [(Fraction(0), (0, 1, 1)), (Fraction(1, 3), (2, 1, 1))]
    )


@pytest.mark.parametrize("origin", DEVICES)
@pytest.mark.parametrize("target", DEVICES)
def test_zero_cell_checkpoint_continues_and_preserves_terminal_fault(
    origin, target, tmp_path
):
    program = observer()
    session = Session(program, device=origin, mode=wp.config.mode)
    first = session.step(until="1", max_events=2)[0]
    path = tmp_path / "empty-state.json"
    session.checkpoint(path)
    continued = Session.restore(path, device=target, mode=wp.config.mode).step(
        until="1"
    )[0]
    assert continued.state == {}
    assert replace(
        continued, records=first.records + continued.records
    ) == run_reference(program, until="1")

    faulting = Session(
        program,
        (Scenario((Drive("1/2", {"accepted": 0}),)),),
        device=origin,
        mode=wp.config.mode,
    )
    fault = faulting.step(until="1")[0]
    faulting.checkpoint(path)
    terminal = Session.restore(path, device=target, mode=wp.config.mode).step(
        until="2"
    )[0]
    assert terminal == replace(fault, records=(fault.fault,))
    assert terminal.fault.state == {}


@pytest.mark.parametrize("device", DEVICES)
def test_zero_cell_empty_calendar_and_case_roundtrip(device, tmp_path):
    program = Program("empty", (), (), ())
    case = Case(program, "1")
    path = tmp_path / "case.json"
    case.save(path)
    assert Case.load(path) == case
    result = Case.load(path).run(device=device, mode=wp.config.mode)[0]
    assert result.status == "horizon" and result.time is None and result.events == 0
    assert result.state == {} and result.records == ()
    assert result.calendar == {"clocks": {}, "timers": {}, "inputs": {}, "cursor": 0}
    session = Session(program, device=device, mode=wp.config.mode)
    session.step(until="1")
    session.checkpoint(path)
    assert (
        Session.restore(path, device=device, mode=wp.config.mode).step(until="2")[0]
        == result
    )


@pytest.mark.parametrize("device", DEVICES)
def test_timer_only_program_needs_no_dummy_cells(device):
    program = Program(
        "timer_notifications",
        (),
        (),
        (
            Region(
                "fire",
                (Observe("wake", (now(),)), Schedule("wake", duration("1/3"))),
                due("wake"),
            ),
        ),
        timers=(Timer("wake", "0"),),
        stop=op("eq", generation("wake"), const(3)),
    )
    expected = run_reference(program, until="1")
    actual = (
        expected
        if device == "reference"
        else run_native(program, until="1", device=device, mode=wp.config.mode)
    )
    assert actual == expected
    assert actual.status == "complete" and actual.time == Fraction(2, 3)
    assert actual.events == 3 and actual.state == {}
    assert actual.calendar["timers"]["wake"] == ((3, 0), 3)
    assert [r.values for r in actual.records if r.kind == "observe"] == [
        (0,),
        (1,),
        (2,),
    ]


def relay_model():
    relay = Component(
        "increment_relay",
        (),
        ("clock",),
        (Port("value", BYTE, "clock"),),
        (Output("out", op("add", inp("value", BYTE), const(1, BYTE)), "clock"),),
        (
            Region(
                "observe", (Observe("relayed", (inp("value", BYTE),)),), due("clock")
            ),
        ),
    )
    consumer = Component(
        "consumer",
        (Cell("seen", BYTE),),
        ("clock",),
        (Port("value", BYTE, "clock"),),
        (),
        (Region("take", (Store("seen", inp("value", BYTE)),), due("clock")),),
    )
    return Model(
        "stateless_composition",
        (Clock("tick"),),
        (Input("external", BYTE),),
        (
            Instance("relay", relay, (("clock", "tick"),)),
            Instance("consumer", consumer, (("clock", "tick"),)),
        ),
        (
            Binding(PortRef("relay", "value"), inp("external", BYTE)),
            Binding(PortRef("consumer", "value"), PortRef("relay", "out")),
        ),
        (
            Transaction(
                "transfer",
                (ActionRef("relay", "observe"), ActionRef("consumer", "take")),
            ),
        ),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_stateless_relay_feeds_stateful_consumer_with_certificate_and_provenance(
    device,
):
    model = relay_model()
    elaboration = elaborate(model)
    assert verify_schedule(elaboration.program, elaboration.certificate)
    assert verify_elaboration(model, elaboration)
    assert tuple(cell.name for cell in elaboration.program.cells) == ("consumer__seen",)
    drives = (Drive("0", {"external": 7}), Drive("1", {"external": 19}))
    expected = run_reference(elaboration.program, until="1", drives=drives)
    actual = (
        expected
        if device == "reference"
        else run_native(
            elaboration.program,
            until="1",
            drives=drives,
            device=device,
            mode=wp.config.mode,
        )
    )
    assert actual == expected
    assert actual.state == {"consumer__seen": (20,)}
    observed = [record for record in actual.records if record.kind == "observe"]
    assert [record.values for record in observed] == [(7,), (19,)]
    for record in observed:
        origin = elaboration.origin(record.location)
        assert (
            origin.instance,
            origin.component,
            origin.action,
            origin.source_path,
        ) == ("relay", "increment_relay", "observe", "body/0")


@pytest.mark.parametrize("device", DEVICES)
def test_composition_of_only_stateless_components_has_empty_state(device):
    model = relay_model()
    relay = model.instances[0]
    model = replace(
        model,
        instances=(relay,),
        bindings=model.bindings[:1],
        transactions=(Transaction("observe", (ActionRef("relay", "observe"),)),),
    )
    elaboration = elaborate(model)
    assert elaboration.program.cells == ()
    assert verify_elaboration(model, elaboration)
    expected = run_reference(
        elaboration.program, until="0", drives=(Drive("0", {"external": 11}),)
    )
    actual = (
        expected
        if device == "reference"
        else run_native(
            elaboration.program,
            until="0",
            drives=(Drive("0", {"external": 11}),),
            device=device,
            mode=wp.config.mode,
        )
    )
    assert actual == expected
    assert actual.state == {} and [
        record.values for record in actual.records if record.kind == "observe"
    ] == [(11,)]
