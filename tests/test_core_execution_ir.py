"""Contracts at the semantic-declaration / native-execution boundary."""

from dataclasses import fields, is_dataclass, replace

from warpsim.core import (
    Bits,
    Cell,
    Clock,
    Program,
    Region,
    Store,
    Require,
    Table,
    const,
    read,
    lookup,
    op,
    lower,
)
from warpsim.core import ir
from warpsim.core import execution_ir as x
from warpsim.core.codegen import Emitter


def walk(value):
    yield value
    if is_dataclass(value):
        for field in fields(value):
            yield from walk(getattr(value, field.name))
    elif isinstance(value, tuple):
        for child in value:
            yield from walk(child)


def example():
    byte = Bits(8)
    return Program(
        "execution_contract",
        (Cell("padding", byte, 3), Cell("destination", byte, 2)),
        (Clock("clock"),),
        (
            Region(
                "write",
                (
                    Store(
                        "destination",
                        op(
                            "select",
                            const(1, ir.BOOL),
                            read("destination", byte),
                            lookup("data", byte, const(99)),
                        ),
                        const(1),
                    ),
                    Require(const(1, ir.BOOL), "checked"),
                ),
            ),
        ),
        tables=(Table("data", byte, 2, (7, 9)),),
    )


def test_execution_ir_resolves_accesses_and_preserves_lazy_fault_order():
    lowered = lower(example())
    executable = x.lower_execution(lowered)
    nodes = tuple(walk(executable))
    assert not any(
        isinstance(
            n,
            (
                ir.Expr,
                ir.Let,
                ir.Store,
                ir.Require,
                ir.Observe,
                ir.If,
                ir.Region,
                ir.Repeat,
            ),
        )
        for n in nodes
    )
    region = executable.schedule[0]
    assert isinstance(region.body[0], x.BeginJournal)
    assert isinstance(region.body[-1], x.Commit)
    statement = region.body[1]
    assert isinstance(statement.body[0], x.CheckIndex)
    assert (statement.body[0].offset, statement.body[0].size) == (3, 2)
    assert isinstance(statement.body[-1], x.CandidateWrite)
    branch = next(n for n in statement.body if isinstance(n, x.Branch))
    # The invalid table lookup must remain in the unselected arm, never
    # speculatively evaluated before the selected state load.
    assert isinstance(branch.then[0], x.CheckedLoad)
    assert branch.then[0].space == "state" and branch.then[0].offset == 3
    assert isinstance(branch.otherwise[0], x.CheckedLoad)
    assert branch.otherwise[0].space == "table" and branch.otherwise[0].index.data == 99
    for node in nodes:
        if hasattr(node, "origin"):
            assert node.origin.location == lowered.locations[node.origin.path][0]
        if isinstance(node, x.Value):
            assert isinstance(node.type, Bits)


def test_native_emission_consumes_execution_ir_not_semantic_statements():
    emitter = Emitter(lower(example()))
    # Once lowered, semantic bodies are no longer read by the native backend.
    # Metadata remains available for the scheduler, storage and diagnostics.
    emitter.p = replace(emitter.p, captures=None, schedule=None, stop=None)
    actual = emitter.generate()
    assert actual == Emitter(lower(example())).generate()
