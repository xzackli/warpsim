# WarpSim core

A finite-state modeling library with exact discrete time, transactional updates,
component composition, and bounded execution on an independent Python interpreter,
native CPU code, or CUDA. All backends consume the same closed representation.

## Install and run

Python 3.12 or newer and a C++17 compiler (`g++`) are required for native CPU
execution. The pinned Warp dependency provides CUDA compilation and device access;
CUDA additionally needs a compatible NVIDIA driver. A requested CUDA backend does
not fall back to CPU. CPU execution does not require a GPU.

```sh
uv sync --locked --extra test --python 3.13
uv run vlsi-sim core example fifo-pipeline /tmp/pipeline.json
uv run vlsi-sim core verify /tmp/pipeline.json --device cpu --output /tmp/verified.json
uv run python -m pytest -q
```

`warpsim-core` exposes the same commands without the `core` prefix. `verify`
compares complete results against the interpreter. Agreement on a modeled fault
is an agreeing execution, not a successful design: inspect each result's status
and fault as well as the report's `matches` value.

## Declare a model

```python
from warpsim.core import (
    Bits, Cell, Clock, Program, Region, Store, Observe,
    const, due, op, read, run_reference, run_native,
)

word = Bits(8)
program = Program(
    "counter",
    (Cell("count", word),),
    (Clock("tick", "1/2"),),
    (Region("increment", (
        Observe("before", (read("count", word),)),
        Store("count", op("add", read("count", word), const(1, word))),
    ), due("tick")),),
)
expected = run_reference(program, until="2", max_events=10)
actual = run_native(program, device="cpu", until="2", max_events=10)
assert actual == expected
assert actual.state["count"] == (5,)
```

Clocks include phase zero and the horizon is inclusive. A region reads its entry
state, stages writes, and commits atomically after all checks pass. Later regions
see preceding commits. Observations above therefore report the old counter value.

## Interfaces

- [CORE.md](CORE.md): types, exact time, transactions, timers, microsteps,
  diagnostics, cases, checkpoints, and supported bounds.
- [MODELING.md](MODELING.md): typed components, ports, ownership, connections,
  atomic groups, explicit ordering, and declaration provenance.
- [EXECUTION.md](EXECUTION.md): batches, resident execution, CPU threading,
  CUDA strategies, inspection, and measurement.
- `warpsim.core.primitives`: declarative ring buffers, arithmetic/filter builders,
  and arbitration utilities. These expand to ordinary semantic operations.
- `examples/`: small queue/filter and traffic-control demonstrations.
- `tests/`: semantic, composition, compilation, continuation, and execution checks.

For a model authored through the Python API, serialize `Case(program, until,
scenarios, max_events)` and use the same CLI to compile, verify, run, or resume it.
Application policies live in declarations or components; the compiler does not
dispatch on application names. Inputs and immutable table data can vary across
scenarios without changing a program's topology.

Passing finite tests is evidence about those runs. It does not prove safety for
all possible inputs, schedules, or parameter values.
