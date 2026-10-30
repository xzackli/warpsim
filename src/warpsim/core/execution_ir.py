"""Typed, ordered execution IR between validated declarations and native code.

This representation contains no semantic expressions or statements. Operands
are typed values, accesses have resolved storage offsets and explicit bounds,
and region writes/observations are candidates until a Commit instruction. Every
instruction retains the declaration path and diagnostic location. Ordering is
intentional: neither eager operands nor potential faults may be reordered.
"""

from dataclasses import dataclass

from . import ir
from .lower import rational


@dataclass(frozen=True)
class Origin:
    path: str
    location: int


@dataclass(frozen=True)
class Value:
    """A literal, input/clock slot, current time, or single-assignment temporary."""

    kind: str
    data: int
    type: ir.Bits


@dataclass(frozen=True)
class Declare:
    result: Value
    origin: Origin


@dataclass(frozen=True)
class Assign:
    result: Value
    value: Value
    origin: Origin


@dataclass(frozen=True)
class CheckedLoad:
    result: Value
    space: str
    offset: int
    size: int
    index: Value
    origin: Origin


@dataclass(frozen=True)
class Compute:
    result: Value
    op: str
    operands: tuple[Value, ...]
    policy: str
    origin: Origin


@dataclass(frozen=True)
class CheckIndex:
    offset: int
    size: int
    index: Value
    origin: Origin


@dataclass(frozen=True)
class CandidateWrite:
    offset: int
    index: Value
    value: Value
    origin: Origin


@dataclass(frozen=True)
class CandidateObservation:
    observation: int
    values: tuple[Value, ...]
    origin: Origin


@dataclass(frozen=True)
class CandidateEvent:
    timer: int
    delay: Value | None
    microstep: Value | None
    origin: Origin


@dataclass(frozen=True)
class FailRequirement:
    details: tuple[Value, ...]
    origin: Origin


@dataclass(frozen=True)
class Branch:
    condition: Value
    then: tuple
    otherwise: tuple
    origin: Origin


@dataclass(frozen=True)
class Statement:
    """Fault-gated scope which establishes the diagnostic location."""

    body: tuple
    origin: Origin


@dataclass(frozen=True)
class BeginJournal:
    origin: Origin


@dataclass(frozen=True)
class Outcome:
    """An explicit diagnostic outcome after validation or a disabled guard."""

    kind: str
    origin: Origin


@dataclass(frozen=True)
class Commit:
    """Publish candidates, then emit the specified successful outcomes."""

    outcomes: tuple
    origin: Origin


@dataclass(frozen=True)
class Region:
    region: int
    guard: tuple
    enabled: Value
    body: tuple
    disabled: Outcome
    origin: Origin


@dataclass(frozen=True)
class Repeat:
    limit: int
    condition_code: tuple
    condition: Value
    body: tuple
    origin: Origin


@dataclass(frozen=True)
class ExecutionProgram:
    captures: tuple
    schedule: tuple
    stop_code: tuple
    stop: Value
    stop_origin: Origin
    temporary_count: int


class _Lowerer:
    def __init__(self, lowered):
        self.l, self.p = lowered, lowered.program
        self.serial = 0
        self.inputs = {v.name: i for i, v in enumerate(self.p.inputs)}
        self.clocks = {v.name: i for i, v in enumerate(self.p.clocks)}
        self.timers = {v.name: i for i, v in enumerate(self.p.timers)}
        self.sources = self.clocks | {
            name: len(self.clocks) + i for name, i in self.timers.items()
        }
        self.cells = {v.name: v for v in self.p.cells}
        self.tables = {v.name: v for v in self.p.tables}

    def origin(self, path):
        return Origin(path, self.l.locations[path][0])

    def temporary(self, type):
        self.serial += 1
        return Value("temporary", self.serial, type)

    def expression(self, e, env, out, origin):
        args, op = e.args, e.op
        if op == "const":
            return Value("literal", args[0], e.type)
        if op == "local":
            return env[args[0]]
        if op == "input":
            return Value("input", self.inputs[args[0]], e.type)
        if op == "due":
            return Value("due", self.sources[args[0]], e.type)
        if op in ("time", "microstep"):
            return Value(op, 0, e.type)
        if op == "duration":
            return Value(
                "duration", int(rational(args[0]) * self.l.denominator), e.type
            )
        if op in ("generation", "due_generation"):
            return Value(op, self.timers[args[0]], e.type)
        result = self.temporary(e.type)
        if op == "select":
            # Only the selected arm executes. Its result is merged explicitly.
            out.append(Declare(result, origin))
            condition = self.expression(args[0], env, out, origin)
            arms = []
            for arm in args[1:]:
                code = []
                value = self.expression(arm, env, code, origin)
                code.append(Assign(result, value, origin))
                arms.append(tuple(code))
            out.append(Branch(condition, *arms, origin))
        elif op in ("read", "lookup"):
            index = self.expression(args[1], env, out, origin)
            table = op == "lookup"
            declaration = (self.tables if table else self.cells)[args[0]]
            offset = (self.l.table_offsets if table else self.l.offsets)[args[0]]
            out.append(
                CheckedLoad(
                    result,
                    "table" if table else "state",
                    offset,
                    declaration.size,
                    index,
                    origin,
                )
            )
        else:
            operands = tuple(self.expression(a, env, out, origin) for a in args)
            out.append(Compute(result, op, operands, e.policy, origin))
        return result

    def statements(self, body, env, path):
        out = []
        for i, statement in enumerate(body):
            key = f"{path}/{i}"
            origin = self.origin(key)
            code = []
            if type(statement) is ir.Let:
                result = self.temporary(statement.value.type)
                # The binding outlives its fault-gated initializer, while all
                # expression temporaries remain inside that initializer scope.
                out.append(Declare(result, origin))
                value = self.expression(statement.value, env, code, origin)
                code.append(Assign(result, value, origin))
                env[statement.name] = result
            elif type(statement) is ir.Store:
                index = self.expression(statement.index, env, code, origin)
                offset = self.l.offsets[statement.target]
                code.append(
                    CheckIndex(offset, self.cells[statement.target].size, index, origin)
                )
                value = self.expression(statement.value, env, code, origin)
                code.append(CandidateWrite(offset, index, value, origin))
            elif type(statement) is ir.Require:
                condition = self.expression(statement.condition, env, code, origin)
                failure = []
                details = tuple(
                    self.expression(x, env, failure, origin) for x in statement.details
                )
                failure.append(FailRequirement(details, origin))
                code.append(Branch(condition, (), tuple(failure), origin))
            elif type(statement) in (ir.Schedule, ir.Cancel):
                delay = step = None
                if type(statement) is ir.Schedule:
                    delay = self.expression(statement.delay, env, code, origin)
                    step = self.expression(statement.microstep, env, code, origin)
                code.append(
                    CandidateEvent(self.timers[statement.timer], delay, step, origin)
                )
            elif type(statement) is ir.Observe:
                values = tuple(
                    self.expression(x, env, code, origin) for x in statement.values
                )
                code.append(
                    CandidateObservation(
                        self.l.observations[statement.name][0], values, origin
                    )
                )
            elif type(statement) is ir.If:
                condition = self.expression(statement.condition, env, code, origin)
                code.append(
                    Branch(
                        condition,
                        self.statements(statement.then, env.copy(), key + "/then"),
                        self.statements(statement.otherwise, env.copy(), key + "/else"),
                        origin,
                    )
                )
            else:
                raise TypeError(f"Unsupported validated statement: {type(statement)}")
            out.append(Statement(tuple(code), origin))
        return tuple(out)

    def region(self, region, env, path):
        origin = self.origin(path)
        guard = []
        enabled = self.expression(region.guard, env, guard, origin)
        body = (
            BeginJournal(origin),
            *self.statements(region.body, env.copy(), path),
            Commit(
                (Outcome("commit", origin), Outcome("observations", origin)), origin
            ),
        )
        return Region(
            self.l.regions[region.name][0],
            tuple(guard),
            enabled,
            body,
            Outcome("disabled", origin),
            origin,
        )

    def generate(self):
        env = {}
        captures = self.statements(self.p.captures, env, "capture")
        schedule = []
        for i, item in enumerate(self.p.schedule):
            path = f"schedule/{i}"
            if type(item) is ir.Region:
                schedule.append(self.region(item, env, path))
            else:
                origin = self.origin(path)
                code = []
                condition = self.expression(item.condition, env, code, origin)
                body = tuple(
                    self.region(r, env, f"{path}/region/{j}")
                    for j, r in enumerate(item.regions)
                )
                schedule.append(
                    Repeat(item.limit, tuple(code), condition, body, origin)
                )
        stop_origin = self.origin("stop")
        code = []
        stop = self.expression(self.p.stop, {}, code, stop_origin)
        return ExecutionProgram(
            captures, tuple(schedule), tuple(code), stop, stop_origin, self.serial
        )


def lower_execution(lowered):
    """Resolve an already validated Lowered program into ordered native actions."""
    return _Lowerer(lowered).generate()
