"""Reusable stateful components with declared clocked ports and atomic actions."""

from .core.ir import (
    Bits,
    BOOL,
    TIME,
    Region,
    Let,
    Observe,
    If,
    const,
    inp,
    local,
    due,
    op,
)
from .core.primitives import Ring
from .modeling import Component, Port, Output


def fifo(*, capacity, value_type=Bits(16, True), identity_type=Bits(32)):
    """Owned FIFO: no fallthrough, old-head full exchange, held output identity.

    Every client uses ports; only this component writes its count and pointers.
    Put/take and both observations belong to the caller's shared transaction.
    """
    q = Ring(
        "queue",
        capacity,
        (("value", value_type), ("id", identity_type), ("time", TIME)),
    )
    valid = op("gt", q.count, const(0, q.count_type))
    ready = op(
        "or",
        op("lt", q.count, const(capacity, q.count_type)),
        op("and", valid, inp("out_ready", BOOL)),
    )
    take, put = local("take", BOOL), local("put", BOOL)
    payload = {
        "value": inp("in_value", value_type),
        "id": inp("in_id", identity_type),
        "time": inp("in_time", TIME),
    }
    return Component(
        "fifo",
        q.cells,
        ("clock",),
        (
            Port("in_valid", BOOL, "clock"),
            Port("in_value", value_type, "clock"),
            Port("in_id", identity_type, "clock"),
            Port("in_time", TIME, "clock"),
            Port("out_ready", BOOL, "clock"),
        ),
        (
            Output("in_ready", ready, "clock"),
            Output("out_valid", valid, "clock"),
            *(
                Output(
                    "out_" + name,
                    op("select", valid, q.front(name), const(0, typ)),
                    "clock",
                )
                for name, typ in q.fields
            ),
        ),
        (
            Region(
                "step",
                (
                    If(
                        take,
                        (
                            Observe(
                                "dequeued",
                                (q.front("id"), q.front("value"), q.front("time")),
                            ),
                        ),
                    ),
                    *q.update(put, take, payload),
                    If(
                        put,
                        (
                            Observe(
                                "enqueued",
                                (payload["id"], payload["value"], payload["time"]),
                            ),
                        ),
                    ),
                ),
                due("clock"),
            ),
        ),
        (
            Let(
                "take",
                op("and", due("clock"), op("and", valid, inp("out_ready", BOOL))),
            ),
            Let(
                "put", op("and", due("clock"), op("and", inp("in_valid", BOOL), ready))
            ),
        ),
    )
