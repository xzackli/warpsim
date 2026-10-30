"""Closed typed transaction IR and shared native execution."""

from .ir import (
    BOOL,
    TIME,
    Bits,
    Cancel,
    Cell,
    Clock,
    Drive,
    Expr,
    If,
    Input,
    Let,
    Observe,
    Program,
    Region,
    Repeat,
    Require,
    Schedule,
    Store,
    Table,
    Timer,
    const,
    due,
    duration,
    generation,
    inp,
    local,
    lookup,
    microstep,
    now,
    op,
    read,
)
from .lower import lower
from .reference import run_reference
from .serialization import Case
from .session import Progress, Scenario, Session

__all__ = [
    "Bits",
    "BOOL",
    "TIME",
    "Expr",
    "const",
    "read",
    "inp",
    "lookup",
    "local",
    "due",
    "now",
    "duration",
    "microstep",
    "generation",
    "op",
    "Cell",
    "Input",
    "Table",
    "Clock",
    "Timer",
    "Schedule",
    "Cancel",
    "Let",
    "Store",
    "Require",
    "Observe",
    "If",
    "Region",
    "Repeat",
    "Program",
    "Drive",
    "lower",
    "run_reference",
    "run_native",
    "Session",
    "Scenario",
    "Progress",
    "Case",
]


def run_native(*args, **kwargs):
    """Load native dependencies only when native execution is requested."""
    from .native import run_native as execute

    return execute(*args, **kwargs)
