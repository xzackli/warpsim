"""Closed, immutable transition language; no host callbacks or model dispatch.

Names are diagnostic labels. Expressions are explicitly typed. Conversion policy
is attached to every cast/arithmetic result, never inferred from native C++.
"""

from dataclasses import dataclass

SCHEMA = "warpsim.core/v2"


@dataclass(frozen=True)
class Bits:
    width: int
    signed: bool = False

    @property
    def minimum(self):
        return -(1 << (self.width - 1)) if self.signed else 0

    @property
    def maximum(self):
        return (1 << (self.width - int(self.signed))) - 1


BOOL = Bits(1)
TIME = Bits(62)


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple
    type: Bits
    policy: str = "checked"


def const(value, type=TIME):
    return Expr("const", (value,), type)


def read(name, type, index=None):
    return Expr("read", (name, const(0) if index is None else index), type)


def inp(name, type):
    return Expr("input", (name,), type)


def lookup(name, type, index):
    return Expr("lookup", (name, index), type)


def local(name, type):
    return Expr("local", (name,), type)


def due(name):
    return Expr("due", (name,), BOOL)


def duration(seconds):
    """An exact rational duration, converted to this run's checked tick lattice."""
    return Expr("duration", (seconds,), TIME)


def now():
    return Expr("time", (), TIME)


def microstep():
    return Expr("microstep", (), TIME)


def generation(timer, *, captured=False):
    """Current or dispatched generation; cancellation does not change due flags."""
    return Expr("due_generation" if captured else "generation", (timer,), TIME)


def op(kind, *args, type=None, policy="checked"):
    if type is None:
        type = (
            BOOL
            if kind in ("eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not")
            else args[1].type
            if kind == "select"
            else args[0].type
        )
    return Expr(kind, tuple(args), type, policy)


@dataclass(frozen=True)
class Cell:
    name: str
    type: Bits
    size: int = 1
    initial: tuple = ()  # empty means all zero


@dataclass(frozen=True)
class Input:
    name: str
    type: Bits
    initial: int = 0


@dataclass(frozen=True)
class Table:
    """Immutable indexed data; omitted values must be supplied at execution."""

    name: str
    type: Bits
    size: int
    values: tuple = ()


@dataclass(frozen=True)
class Clock:
    name: str
    period: str = "1"
    phase: str = "0"  # phase zero is dispatched


@dataclass(frozen=True)
class Timer:
    """One owned pending-event slot. Re-arming replaces its future occurrence."""

    name: str
    initial: str | None = None
    microstep: int = 0


@dataclass(frozen=True)
class Schedule:
    timer: str
    delay: Expr
    microstep: Expr = const(0)


@dataclass(frozen=True)
class Cancel:
    timer: str


@dataclass(frozen=True)
class Let:
    name: str
    value: Expr


@dataclass(frozen=True)
class Store:
    target: str
    value: Expr
    index: Expr = const(0)


@dataclass(frozen=True)
class Require:
    condition: Expr
    message: str
    details: tuple = ()


@dataclass(frozen=True)
class Observe:
    name: str
    values: tuple = ()


@dataclass(frozen=True)
class If:
    condition: Expr
    then: tuple
    otherwise: tuple = ()


@dataclass(frozen=True)
class Region:
    name: str
    body: tuple
    guard: Expr = const(1, BOOL)


@dataclass(frozen=True)
class Repeat:
    """While condition, run ordered regions; a still-true bound is a fault."""

    name: str
    condition: Expr
    limit: int
    regions: tuple


@dataclass(frozen=True)
class Program:
    name: str
    cells: tuple
    clocks: tuple
    schedule: tuple
    inputs: tuple = ()
    captures: tuple = ()  # scalar values captured once at group entry
    tables: tuple = ()
    stop: Expr = const(0, BOOL)  # checked after each successfully executed group
    timers: tuple = ()
    max_microstep: int = 1024


@dataclass(frozen=True)
class Drive:
    time: str
    values: dict
