"""Closed semantic workloads for execution-strategy experiments.

These are ordinary models, not executor dispatch keys. ``size`` means array
width for map/stencil/sparse, dependent arithmetic depth for chain, and is only
a recorded sweep label for the small control/timer/skew workloads. Diverse
batches cycle through sixteen deterministic scenarios; identical batches repeat
one scenario. No randomness or native implementation appears in a fixture.
"""

from .core import (
    BOOL,
    Bits,
    Cancel,
    Case,
    Cell,
    Clock,
    Drive,
    If,
    Input,
    Let,
    Observe,
    Program,
    Region,
    Require,
    Scenario,
    Schedule,
    Store,
    Timer,
    const,
    due,
    duration,
    inp,
    local,
    op,
    read,
)

WORD = Bits(32)
COUNT = Bits(32)
FIXTURES = {
    "control": "Small state with data-dependent branches and changing inputs",
    "sparse": "Large array with one indirect read/write per event",
    "map": "Independent constant-address arithmetic updates across an array",
    "stencil": "Constant-address three-point stencil within one atomic region",
    "chain": "Dependent arithmetic Lets with a single final state write",
    "timers": "Exact rational clocks, cancellation, and ordered microsteps",
    "skew": "Mixed early completion, expected faults, and long survivors",
}


def _binary(kind, a, b):
    return op(kind, a, b, policy="wrap")


def _counter():
    return Region(
        "counter",
        (
            Store("steps", op("add", read("steps", COUNT), const(1, COUNT))),
            Observe("step", (read("steps", COUNT),)),
        ),
        due("tick"),
    )


def _program(name, cells, regions, *, inputs=(), timers=(), period="1"):
    return Program(
        f"campaign_{name}",
        (Cell("steps", COUNT), *cells),
        (Clock("tick", period),),
        tuple(regions),
        inputs=(Input("target", COUNT), Input("seed", WORD), *inputs),
        stop=op("ge", read("steps", COUNT), inp("target", COUNT)),
        timers=timers,
        max_microstep=4,
    )


def _declare(name, size):
    value, seed = read("value", WORD), inp("seed", WORD)
    if name in ("control", "skew"):
        body = (
            If(
                op("eq", op("mod", seed, const(3, WORD)), const(0, WORD)),
                (Store("value", _binary("add", value, seed)),),
                (
                    If(
                        op("lt", value, const(10000, WORD)),
                        (
                            Store(
                                "value",
                                _binary(
                                    "add", _binary("mul", value, const(17, WORD)), seed
                                ),
                            ),
                        ),
                        (Store("value", _binary("sub", value, seed)),),
                    ),
                ),
            ),
        )
        inputs = ()
        if name == "skew":
            inputs = (Input("fail", BOOL), Input("fail_at", COUNT))
            # The staged write must roll back while prior event groups survive.
            body += (
                Require(
                    op(
                        "or",
                        op("not", inp("fail", BOOL)),
                        op("lt", read("steps", COUNT), inp("fail_at", COUNT)),
                    ),
                    "expected skew fixture failure",
                    (read("steps", COUNT), value),
                ),
            )
        return _program(
            name,
            (Cell("value", WORD, initial=(1,)),),
            (Region("update", body, due("tick")), _counter()),
            inputs=inputs,
        )
    if name == "sparse":
        pointer = read("pointer", COUNT)
        return _program(
            name,
            (Cell("array", WORD, size), Cell("pointer", COUNT)),
            (
                Region(
                    "touch",
                    (
                        Store(
                            "array",
                            _binary("add", read("array", WORD, pointer), seed),
                            pointer,
                        ),
                        Store(
                            "pointer",
                            op(
                                "mod",
                                op("add", pointer, const(17, COUNT)),
                                const(size, COUNT),
                            ),
                        ),
                    ),
                    due("tick"),
                ),
                _counter(),
            ),
        )
    if name in ("map", "stencil"):
        bias = read("bias", WORD)
        stores = []
        for i in range(size):
            current = read("array", WORD, const(i))
            if name == "map":
                expression = _binary(
                    "add", _binary("mul", current, const(1664525, WORD)), bias
                )
            else:
                expression = _binary(
                    "add",
                    _binary(
                        "add",
                        read("array", WORD, const((i - 1) % size)),
                        _binary("mul", current, const(2, WORD)),
                    ),
                    _binary("add", read("array", WORD, const((i + 1) % size)), bias),
                )
            stores.append(Store("array", expression, const(i)))
        return _program(
            name,
            (Cell("array", WORD, size, tuple(range(size))), Cell("bias", WORD)),
            (Region("array_update", tuple(stores), due("tick")), _counter()),
        )
    if name == "chain":
        statements = []
        previous = value
        for i in range(size):
            expression = _binary(
                "add", _binary("mul", previous, const(1664525, WORD)), seed
            )
            statements.append(Let(f"v{i}", expression))
            previous = local(f"v{i}", WORD)
        return _program(
            name,
            (Cell("value", WORD, initial=(1,)),),
            (
                Region("chain", (*statements, Store("value", previous)), due("tick")),
                _counter(),
            ),
        )
    if name == "timers":
        return _program(
            name,
            (Cell("value", WORD),),
            (
                Region(
                    "sources",
                    (Observe("sources", (due("tick"), due("pulse"), due("delayed"))),),
                ),
                Region(
                    "arm",
                    (
                        Schedule("pulse", duration("0"), const(1)),
                        Schedule("delayed", duration("1/6")),
                    ),
                    due("tick"),
                ),
                Region(
                    "cancel",
                    (Cancel("delayed"),),
                    op(
                        "and",
                        due("tick"),
                        op(
                            "eq",
                            op("mod", read("steps", COUNT), const(2, COUNT)),
                            const(0, COUNT),
                        ),
                    ),
                ),
                Region(
                    "pulse",
                    (Store("value", _binary("add", value, seed)), Observe("pulse")),
                    due("pulse"),
                ),
                Region(
                    "delayed",
                    (Store("value", _binary("add", value, const(1, WORD))),),
                    due("delayed"),
                ),
                _counter(),
            ),
            timers=(Timer("pulse"), Timer("delayed")),
            period="1/3",
        )
    raise ValueError(f"Unknown campaign fixture: {name}")


def build_case(name, *, batch=32, size=64, events=64, distribution="diverse"):
    """Build a deterministic, bounded Case entirely through public declarations.

    ``events`` is the maximum number of periodic ticks until intentional stop;
    timer fixtures also execute intermediate event groups. The independent
    event-group budget is generous enough to include those groups. Skew has
    1/16 long-lived scenarios, 4/16 faults, and otherwise short completions.
    """
    if name not in FIXTURES:
        raise ValueError(f"Unknown campaign fixture: {name}")
    for value, low, high, label in (
        (batch, 1, 65536, "batch"),
        (events, 1, 1000000, "events"),
        (size, 1, 65534, "size"),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    if name in ("map", "stencil") and size > 1024:
        raise ValueError("Array-region size must be at most 1024")
    if name == "chain" and size > 1023:
        raise ValueError("Chain depth must be at most 1023")
    if distribution not in ("identical", "diverse"):
        raise ValueError("distribution must be identical or diverse")
    program = _declare(name, size)
    scenarios = []
    for lane in range(batch):
        variant = 15 if distribution == "identical" else lane % 16
        seed = 3 + variant
        target = events
        values = {"target": target, "seed": seed}
        if name == "skew":
            target = events if variant == 15 else min(events, 1 + variant % 4)
            values.update(
                target=target, fail=int(variant % 4 == 2), fail_at=max(0, target - 1)
            )
        drives = [Drive("0", values)]
        # Vary held input values over time without changing the common event
        # lattice; all generated input updates coincide with existing ticks.
        if name == "control" and events > 2:
            drives.append(Drive(str(events // 2), {"seed": seed * 13}))
        scenarios.append(
            Scenario(
                tuple(drives),
                initial={"bias": seed} if name in ("map", "stencil") else None,
            )
        )
    return Case(
        program,
        str(events),
        tuple(scenarios),
        max_events=events * (4 if name == "timers" else 1) + 8,
        trace_cells=("steps",),
    ).validate()
