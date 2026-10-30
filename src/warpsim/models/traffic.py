"""Composed traffic signals with timed phases and sampled emergency control.

The controller owns its phase and timers; two reusable heads own their colors;
an independent interlock consumes only their public outputs. All behavior
elaborates through the public modeling and semantic-core APIs.
"""

from fractions import Fraction

from warpsim.core import (
    BOOL,
    Bits,
    Cancel,
    Cell,
    Clock,
    If,
    Input,
    Let,
    Observe,
    Region,
    Require,
    Schedule,
    Store,
    Timer,
    const,
    due,
    duration,
    generation,
    inp,
    local,
    op,
    read,
)
from warpsim.modeling import (
    ActionRef,
    Binding,
    Component,
    Elaboration,
    Instance,
    Model,
    Output,
    Port,
    PortRef,
    Transaction,
    elaborate,
)


PHASE = Bits(3)
COLOR = Bits(2)


def _positive_duration(value, name):
    if type(value) is not str:
        raise ValueError(f"{name} must be a positive exact rational string")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be a positive exact rational string") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive exact rational string")
    return str(parsed)


def traffic_controller(*, green="2", amber="1/2", clearance="1/3"):
    """Six timed phases; emergency changes are sampled on the formal clock.

    Phases are 0 NS green, 1 NS amber, 2 clearance toward EW, 3 EW green,
    4 EW amber, and 5 clearance toward NS. Reset starts at phase 5. An asserted
    emergency cancels the phase timer and holds phase 5; release starts a fresh
    clearance interval. Durations are positive exact rational strings.

    Port values are sampled at group entry. Separate publish and audit
    microsteps expose the new phase to the heads, then their colors to the
    interlock. Timer-generation checks suppress a switch cancelled or replaced
    by the earlier emergency transaction at the same physical time.
    """
    green = _positive_duration(green, "green")
    amber = _positive_duration(amber, "amber")
    clearance = _positive_duration(clearance, "clearance")
    phase = read("phase", PHASE)
    latched = read("latched", BOOL)
    emergency = inp("emergency", BOOL)
    next_phase = local("next_phase", PHASE)

    def phase_is(value, number):
        return op("eq", value, const(number, PHASE))

    publish_and_audit = (
        Schedule("publish", duration("0"), const(1)),
        Schedule("audit", duration("0"), const(2)),
    )
    next_delay = op(
        "select",
        op("or", phase_is(next_phase, 0), phase_is(next_phase, 3)),
        duration(green),
        op(
            "select",
            op("or", phase_is(next_phase, 1), phase_is(next_phase, 4)),
            duration(amber),
            duration(clearance),
        ),
    )
    return Component(
        "traffic_controller",
        (Cell("phase", PHASE, initial=(5,)), Cell("latched", BOOL)),
        ("sample",),
        (Port("emergency", BOOL, "sample"),),
        (
            Output("ns_green", phase_is(phase, 0), "sample"),
            Output("ns_amber", phase_is(phase, 1), "sample"),
            Output("ew_green", phase_is(phase, 3), "sample"),
            Output("ew_amber", phase_is(phase, 4), "sample"),
            Output("publish", due("publish"), "sample"),
            Output("audit", due("audit"), "sample"),
        ),
        (
            Region(
                "emergency",
                (
                    Store("latched", emergency),
                    Store("phase", const(5, PHASE)),
                    If(
                        emergency,
                        (Cancel("switch"),),
                        (Schedule("switch", duration(clearance)),),
                    ),
                    *publish_and_audit,
                    Observe("emergency", (emergency,)),
                ),
                op("and", due("sample"), op("ne", emergency, latched)),
            ),
            Region(
                "advance",
                (
                    Require(op("le", phase, const(5, PHASE)), "invalid signal phase"),
                    Let(
                        "next_phase",
                        op(
                            "select",
                            phase_is(phase, 5),
                            const(0, PHASE),
                            op("add", phase, const(1, PHASE)),
                        ),
                    ),
                    Store("phase", next_phase),
                    Schedule("switch", next_delay),
                    *publish_and_audit,
                    Observe("phase", (next_phase,)),
                ),
                op(
                    "and",
                    due("switch"),
                    op(
                        "and",
                        op("not", latched),
                        op(
                            "eq",
                            generation("switch", captured=True),
                            generation("switch"),
                        ),
                    ),
                ),
            ),
        ),
        timers=(
            Timer("switch", clearance),
            Timer("publish", "0", 1),
            Timer("audit", "0", 2),
        ),
    )


def signal_head():
    """A reusable red/amber/green head with its own checked display state.

    Inputs are green, amber, and apply. Public outputs are is_green, is_amber,
    and color. Color codes are 0 red, 1 amber, and 2 green. Both asserted color
    requests fault atomically instead of choosing an implicit priority.
    """
    green, amber = inp("green", BOOL), inp("amber", BOOL)
    color = op(
        "select",
        green,
        const(2, COLOR),
        op("select", amber, const(1, COLOR), const(0, COLOR)),
    )
    stored = read("color", COLOR)
    return Component(
        "signal_head",
        (Cell("color", COLOR),),
        ("sample",),
        tuple(Port(name, BOOL, "sample") for name in ("green", "amber", "apply")),
        (
            Output("is_green", op("eq", stored, const(2, COLOR)), "sample"),
            Output("is_amber", op("eq", stored, const(1, COLOR)), "sample"),
            Output("color", stored, "sample"),
        ),
        (
            Region(
                "display",
                (
                    Require(
                        op("not", op("and", green, amber)), "conflicting signal colors"
                    ),
                    Store("color", color),
                    Observe("color", (color,)),
                ),
                inp("apply", BOOL),
            ),
        ),
    )


def interlock_monitor():
    """Check the heads' public green indications and count successful audits."""
    ns, ew = inp("ns_green", BOOL), inp("ew_green", BOOL)
    counter = Bits(32)
    checks = read("checks", counter)
    return Component(
        "interlock_monitor",
        (Cell("checks", counter),),
        ("sample",),
        tuple(Port(name, BOOL, "sample") for name in ("ns_green", "ew_green", "audit")),
        (Output("checks", checks, "sample"),),
        (
            Region(
                "check",
                (
                    Require(op("not", op("and", ns, ew)), "simultaneous green signals"),
                    Store("checks", op("add", checks, const(1, counter))),
                    Observe("checked", (ns, ew)),
                ),
                inp("audit", BOOL),
            ),
        ),
    )


def traffic_signals(*, green="2", amber="1/2", clearance="1/3") -> Elaboration:
    """Elaborate a controller, two shared head instances, and an interlock.

    The external BOOL input ``emergency`` is sampled every 1/3 time unit.
    Controller timers use rational physical time, publication microstep 1,
    and audit microstep 2. The two heads share one display transaction, so a
    failure in either head rolls back both displays.
    """
    head = signal_head()
    instances = (
        Instance(
            "controller",
            traffic_controller(green=green, amber=amber, clearance=clearance),
            (("sample", "sample"),),
        ),
        Instance("ns", head, (("sample", "sample"),)),
        Instance("ew", head, (("sample", "sample"),)),
        Instance("monitor", interlock_monitor(), (("sample", "sample"),)),
    )
    bindings = [Binding(PortRef("controller", "emergency"), inp("emergency", BOOL))]
    for direction in ("ns", "ew"):
        for color in ("green", "amber"):
            bindings.append(
                Binding(
                    PortRef(direction, color),
                    PortRef("controller", f"{direction}_{color}"),
                )
            )
        bindings.append(
            Binding(PortRef(direction, "apply"), PortRef("controller", "publish"))
        )
        bindings.append(
            Binding(
                PortRef("monitor", f"{direction}_green"), PortRef(direction, "is_green")
            )
        )
    bindings.append(
        Binding(PortRef("monitor", "audit"), PortRef("controller", "audit"))
    )
    model = Model(
        "traffic_signals",
        (Clock("sample", "1/3"),),
        (Input("emergency", BOOL),),
        instances,
        tuple(bindings),
        (
            Transaction("emergency", (ActionRef("controller", "emergency"),)),
            Transaction("advance", (ActionRef("controller", "advance"),)),
            Transaction(
                "display", (ActionRef("ns", "display"), ActionRef("ew", "display"))
            ),
            Transaction("audit", (ActionRef("monitor", "check"),)),
        ),
        (("emergency", "advance"), ("advance", "display"), ("display", "audit")),
    )
    return elaborate(model)
