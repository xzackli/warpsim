# Transition semantics and execution

## Run the example

From the installed project environment:

```sh
.venv/bin/python examples/queued_fir.py --device cpu --output results/core-example-cpu.json
.venv/bin/python examples/queued_fir.py --device cuda:0 --output results/core-example-cuda.json
.venv/bin/python -m pytest -q tests/test_core.py
```

The example verifies every commit, disabled region, successful observation and
fault against the independent interpreter. Its JSON includes the model, input
schedule, model hash, exact time unit, complete trace and terminal state. CUDA
requests use the existing pinned Warp native adapter and fail if unavailable.

```python
from warpsim.core import Drive, run_reference, run_native
from warpsim.models.queued_fir import queued_fir

model = queued_fir(capacity=3, producer_period="1/2", consumer_period="2/3")
options = dict(
    until="3",
    drives=(Drive("0", {"valid": 1, "value": 3, "id": 7}),
            Drive("1/2", {"valid": 0})),
)
reference = run_reference(model, **options)
native = run_native(model, device="cpu", **options)
assert reference == native
```

The queued FIR composes the reusable `Ring` and `fir` declaration builders with
held-output control and a repair port. The ring permits full exchange and forbids
empty fallthrough. The edge region samples the old head/history/output, computes
the filter, and atomically applies queue, memory, output and observation changes.
A subsequent repair region sees committed state and uses a captured arbitration
grant. Invalid losing repairs are not evaluated. A repair failure retains the
successful edge. Capacities and periods are parameters; changing them requires
no backend changes. A large history uses the same bounded write journal
as the two-element case.

## Language and execution contract

- `Bits(1..63, signed=...)` describes finite values. `BOOL` is unsigned one bit.
  Cells are fixed arrays; size-one arrays represent scalars. All storage starts
  initialized. Named fields of products can be represented as parallel cells.
- Expressions are closed data: constants, state/input/local reads, clock due
  flags, time, comparisons, Boolean operations, lazy select, casts, add/subtract,
  multiply, division/remainder and explicit power-of-two rounding. Division
  truncates toward zero; remainder has the dividend's sign. Division by zero
  faults, and an unrepresentable result follows its conversion policy. Rounding
  supports ties-to-even, ties-away, floor and toward-zero. Arithmetic results explicitly
  choose checked, wrapping or saturating conversion. Lowering rejects operations
  whose possible intermediate values exceed the signed 64-bit native carrier.
  Wide unsupported multiplication is rejected even with a narrowing result.
- `Table` declares immutable indexed data. An empty values tuple requires a
  runtime table; a constant table cannot be overridden. `lookup` checks bounds
  before loading. Table values do not participate in state snapshots or writes.
- `Region` evaluates its guard and body from its entry state. `Store` stages a
  write; it never forwards implicitly to a later read. `Let` provides explicit
  local values. Every active bounds, arithmetic, assertion and alias check must
  pass before writes and observations publish. Equal duplicate writes conflict.
  Guard-false regions do not evaluate their bodies.
- `Program.captures` contains scalar `Let` bindings evaluated once at event-group
  entry. They remain available to subsequent regions. Region-local bindings do
  not escape their scope. Cross-region historical arrays are unsupported; capture
  the required elements explicitly. Native snapshots require no whole-array copy.
- Regions execute in declared order. Later regions see earlier commits.
  `Repeat` rechecks its condition after the ordered body and retains each earlier
  commit if a later iteration fails. A still-true condition at the repeat limit
  produces `loop_limit`, including a loop whose body makes no progress.
- `Program.stop` evaluates from committed state after each successful group. A
  true condition returns `complete`; a fault in the condition preserves the
  committed group. Captured/region-local values are outside its scope.
- `If` and `select` evaluate only the selected branch. Boolean `and`/`or` are
  eager: put potentially invalid operations in guarded statements or selects.
  Operands and statements are evaluated left to right. First failure wins.
- `Observe` values are staged in statement order and published after the region's
  commit. Trace records carry exact time, microstep, repeat iteration, region, statement path, selected committed
  state, observation payload, and first-failure kind/message/target/index.
  `Require.details` supplies up to eight diagnostic operands; these are evaluated
  only for a false condition, before publishing its fault.

Native generation now consumes `core.execution_ir`, with typed resolved operands,
checked accesses, candidate writes, lazy control, commits and outcomes. It does
not interpret semantic expression/statement nodes during emission. The independent
reference still evaluates the semantic IR. Component elaboration, schedule
certificates, declaration provenance and the FIFO-chain example are described in
[MODELING.md](MODELING.md).

Periodic clocks use exact rational period/phase strings. Phase zero is dispatched.
Each group gathers all clocks, timers and input changes at its minimum `(time, microstep)` tag. Drives
are applied before captures; their values remain held until changed. Multiple
changes at one time must be combined into one `Drive`. Input changes also create
groups when no clock is due, so a clock-specific region must declare a due guard.
Clock occurrences are consumed even when every region is disabled.

The interpreter uses rational occurrence arithmetic. Native code uses a checked
integer lattice formed from all clock and drive denominators. `Result.tick_seconds`
records that lattice; `Expr("time", (), TIME)` returns its tick count. `TIME`
payloads are integers in this run-specific unit, so callers must encode timestamp
payloads accordingly; trace tags themselves are returned as exact `Fraction`s.
The horizon is inclusive. A final event that exactly uses the budget can still
finish at the horizon; `budget` means a further due event remains unprocessed.
Faults retain the consumed event group and the earlier committed prefix.

## Implementation boundaries

`ir.py` defines the semantic data. `lower.py` validates it, records conservative
read/write/observation effects (including guard and index reads), assigns storage
and diagnostic identities, and bounds journals/logs. It preserves declaration
order and performs no effect-based scheduling. `reference.py` interprets the
semantic tree with Python integers and copied snapshots. `codegen.py` specializes
it to native checks, sparse candidate stores, publication and control flow.
`native.py` reuses `native_runtime.py` for compilation and CPU/CUDA launch.

Both backends return the same public records. The interpreter and emitter share
validation and record shapes, but do not share expression/transition evaluation.
The independent deque test supplies an additional semantic witness for the first
composition. Deliberately broken early-store and missing-alias-check emitters are
detected by the differential tests. This is testing, not a compiler proof.

## Timers and semantic microsteps

`Timer(name, initial=None, microstep=0)` declares one bounded pending slot. An
initial rational time dispatches once; a missing time starts inactive.
`Schedule(name, delay, microstep=const(0))` replaces the future occurrence and
`Cancel(name)` clears it. Both effects share the region's journal and rollback
boundary with state and observations. Duplicate selected effects on one slot
conflict, including a schedule followed by a cancel. Timer generations increment
only on committed schedule/cancel, with overflow checked.

Use `duration("1/3")` for exact delay constants, `now()` for integer ticks, and
`microstep()` for the semantic microstep. A zero delay requires a strictly later
microstep; inserting at/before the frontier faults. `Program.max_microstep`
(default 1024, maximum 1,000,000) bounds same-time activity. New physical-time
events normally use microstep zero. Compiler instructions never create microsteps.

Dispatch consumes due occurrences before the handlers execute, including a
failing/disabled handler. `due(name)` stays fixed for the group, even if an earlier
region cancels that timer. `generation(name, captured=True)` supplies the dispatched
generation; compare it with `generation(name)` when the declared policy should
suppress a captured event after cancellation. A later fault preserves all prior
commits and the consumed calendar frontier.

## Batches, continuation, checkpoints and replay

```python
from warpsim.core import Session, Scenario
from warpsim.models.fifo_pipeline import fifo_pipeline

program = fifo_pipeline().program
session = Session(program, (Scenario(), Scenario()), device="cuda:0",
                  trace_mode="summary")
partial = session.step(until="100", max_events=10)
session.checkpoint("/tmp/pipeline.checkpoint.json")
continued = Session.restore("/tmp/pipeline.checkpoint.json", device="cpu")
result = continued.run(until="100", max_events=1000)
```

`step` executes at most 100,000 additional event groups per lane. `run` accepts a
larger finite campaign budget and uses bounded chunks. Results contain cumulative
event counts, complete final state/calendar, and records for the current call.
A terminal fault/completion lane returns its existing terminal result on later
calls. Horizons must be nondecreasing; horizon and budget outcomes are resumable.
A native session keeps state, external inputs and immutable tables resident.
For summary campaigns, `advance`, compact `status`, final `inspect`, active-ID
compaction, native multicore CPU execution, and conservative cooperative CUDA
lowering are described in [Execution API](EXECUTION.md).
All lanes share topology and an exact lattice including all their drive times.
`time_denominator` can explicitly retain a finer lattice for replay; exported
single-scenario failures preserve the original batch's lattice and TIME payloads.

Full traces include commits/disabled attempts/observations/faults. Observation
mode retains observations and faults; summary retains only the first fault.
Summary runs do not allocate a trace proportional to event count. `trace_cells`
selects diagnostic state without affecting complete final state. The packed
buffer budget is checked before native allocation; Python result objects and
compiler memory are additional host costs. Full campaign output is budgeted for
the entire campaign, not just one chunk.

Checkpoints use versioned JSON data constructors, checksums and validated logical
state/calendar. They preserve pending timer tags/generations, clock occurrences,
input cursor/held values, terminal diagnostics, program and original stimuli.
Restore rejects inconsistent frontiers/diagnostics before loading native code.
This validates checkpoint structure and integrity, not reachability of arbitrary
edited states. No pickle or user-named imports are accepted. Checkpoints retain
a terminal fault, not previous successful trace history.

`Case.save/load/run` provides self-contained closed-program replay artifacts.
`vlsi-sim core verify` compares reference and generated results; matching failures
remain failures of the model. `compile` exports source and a semantic-path/line map.
The core uses the caller's declared model time unit. Physical unit conversion
is the responsibility of the model author.

State may be empty and is bounded to 65,536 scalar cells, each region to 4,096 candidate stores and
1,024 observations, and each repeated transaction to 1,024 rounds. Model/source
collections and timer slots are bounded at 256. Immutable tables are bounded to
16,777,216 values. Unsupported wide arithmetic, historical array views and
instantaneous wiring cycles fail validation. Product records use parallel named
cells/ports; unrestricted historical snapshots, inferred arbitration, hierarchical
submodels, dedicated enum/refinement types and rejection-and-continue are outside
the current language. CUDA Graph capture and formal proof are outside this interface.
Agreement on bounded runs does not establish universal safety or performance.

