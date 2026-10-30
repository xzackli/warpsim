# Component modeling and the execution compiler

The executable path is now:

```text
Component declarations, typed ports, connections, atomic groups, order constraints
    -> modeling.elaborate: contracts, ownership, cycles, provenance, certificate
    -> core.ir.Program: closed semantic expressions and transactions
       -> core.reference: independent semantic interpreter
       -> core.lower: types, bounds, effects, storage and diagnostic IDs
       -> core.execution_ir: typed operands, checked loads, candidate writes,
                             branches, loops, commit and outcome instructions
       -> core.codegen: common C++ body with CPU and CUDA entry points
```

The native emitter no longer walks semantic expressions or statement bodies.
The execution IR resolves storage offsets and retains source paths and diagnostic
IDs. Candidate writes remain separate from commit; lazy branches and first-fault
order remain explicit. Tests check that neither semantic nodes nor component
callbacks survive this boundary. The calendar handles exact clocks, input changes and transactional one-shot timer
slots. Tagged microsteps, cancellation generations and portable event-boundary
continuation are specified in [CORE.md](CORE.md).

## Run a composition

```python
from warpsim.core import Drive, run_native, run_reference
from warpsim.models.fifo_pipeline import fifo_pipeline
from warpsim.modeling import verify_schedule

composition = fifo_pipeline((2, 3, 1))
verify_schedule(composition.program, composition.certificate)
drives = (
    Drive("0", {"valid": 1, "value": -7, "id": 10, "source_time": 0}),
    Drive("1", {"valid": 0}),
)
expected = run_reference(composition.program, until="8", drives=drives)
actual = run_native(composition.program, until="8", drives=drives, device="cuda:0")
assert actual == expected
assert any(r.observation == "stage2__dequeued" and r.values == (10, -7, 0)
           for r in actual.records)
```

Use `device="cpu"` for native CPU execution. The pipeline declaration only creates
instances, connects public ports and names an atomic transaction. It does not
modify the execution backend. Tests use an independent deque oracle for both
one-slot FIFOs and a chain with unequal capacities under stalls and full exchange.

`warpsim.components.fifo` owns its ring state and exposes typed valid/value/ID/time
and ready ports. Its declared policy has no fallthrough, permits full exchange
of the old head, and retains the output token until accepted. All participants'
transfers and observations belong to the declared transaction. A failure in any
participant discards every candidate in that group.

## Composition contract

- A `Component` declares private cells/tables/timers, formal clocks, input `Port`s,
  `Output` expressions, captures and named semantic `Region` actions. Its version
  and declaration hash are retained. Component-local validation forbids an action
  from accessing another instance's private state.
- An `Instance` binds every formal clock explicitly. A `Binding` connects an
  output to an input, or supplies a closed expression over external model inputs.
  Input ports require exactly one driver. Connected widths, signedness and clock
  domains must match. A domain crossing needs an explicit component contract;
  the elaborator does not synthesize synchronization.
- Ports have a deliberately narrow **group-entry sampling** contract. Bindings
  are captured in their declared order at every dispatched event group, followed
  by component captures in instance/declaration order. These expressions must
  be valid when sampled; use lazy expressions for data behind invalid tags.
  Unconnected output expressions are checked structurally but are not evaluated.
  Actions still declare their activation guards explicitly.
- Output dependencies through input bindings must be acyclic. Structural wiring
  cannot silently introduce a combinational loop. A dependency through owned
  state breaks the combinational path.
- A `Transaction` lists participant actions in explicit order. Their guards are
  checked left-to-right with short-circuiting: the first disabled participant
  disables the group. Their bodies share one entry-state view and one rollback
  boundary. Action-local values stay scoped. Multiple selected writes to one
  owned cell still undergo the core's runtime alias check.
- Separate transactions observe preceding committed state through local reads.
  Their ports remain captured from group entry. `Model.before` must imply a unique
  transaction order. The prototype conservatively rejects ambiguous order even
  for disjoint writes, because failure and observations are ordering effects.
  It never uses instance iteration order as an arbitration policy.
- A transaction may declare `repeat=PortRef(...)` (or a closed external
  expression) and a finite `limit`. Its condition reads current committed state;
  each round has a separate commit and diagnostic iteration. Entry-sampled ports
  and captures retain their group view. Timer names and their generation reads
  are scoped to each component instance; formal clocks remain explicitly bound.
- A stop output is probed from committed state after a successful group. It does
  not reuse entry port captures. The underlying core's completion/fault rules
  remain unchanged.

`Elaboration.origins` maps generated semantic statement paths back to component
instance, definition/version, action and declaration path. Use
`composition.origin(result.fault.location)` for a failed statement. This is
declaration provenance, not yet a complete source-file/span or replay format.

`verify_schedule` checks the expanded program's explicit order, effects and hash
against its certificate. `verify_elaboration(model, result)` additionally
re-elaborates declarations to check contract metadata and provenance. These are
structural consistency checks, not formal refinement proofs.

## Compiler partitioning

Compile-time arithmetic policy and width are now specialized directly, instead
of forcing a generic conversion helper into every operation. Each fault-gated
execution statement becomes an immediately invoked function scope. CUDA debug builds keep these functions separate. Release does so when the
execution IR exceeds 2,048 temporaries, bounding optimization regions for large
compositions; smaller release programs remain inlined. CPU compilers can inline
the functions. References carry the
same journal, locals and fault state through these calls. This bounds individual
compiler optimization regions without moving a commit or changing evaluation
order. Total temporary count is not claimed to have decreased.

## Supported modeling class

The execution language describes a bounded modeling class, not a list of
supported devices. Its primitives have meanings independent of any application:

| Shared mechanism | General semantic purpose |
| --- | --- |
| Finite typed cells, tables and expressions | Bound storage, indexed access and arithmetic behavior; zero owned cells are valid. |
| Exact clocks, input events and timer slots | Define when transitions run, including replaceable/cancellable deadlines. |
| Regions and ordered transactions | Define visibility, rollback and the committed prefix after failure. |
| Bounded repetition | Express repeated transitions with an explicit termination bound. |
| Ports, bindings and component ownership | Compose declarations without private-state access or implicit wiring order. |
| Observations, cases and checkpoints | Inspect and continue the same closed representation on each backend. |

Application policies belong in declarations built from these mechanisms.
A new core operation needs a general semantic contract for evaluation,
visibility, failure, and reference/native behavior.

`models/traffic.py` is a separate architectural acceptance case: a timed traffic
controller, two instances of a signal-head component, and an independent
interlock communicate through ordinary typed ports. Emergency sampling cancels
or replaces a pending phase deadline. Explicit publication and audit microsteps
make the committed controller phase visible to the heads and then to the
interlock. Both heads share one atomic transaction. Phase durations and the
emergency policy live entirely in the model; no traffic operation or dispatch
was added to the compiler, execution IR, session, checkpoint, or CLI.

```sh
uv run python examples/traffic_signals.py /tmp/signals.json
uv run vlsi-sim core verify /tmp/signals.json --device cuda:0 --output /tmp/signals.verify.json
uv run vlsi-sim core compile /tmp/signals.json --output /tmp/signals.cpp
```

`tests/test_modeling_traffic.py` checks hand-calculated phase/clearance times,
coincident emergency cancellation, changed durations, provenance/source maps,
and all reference/CPU/CUDA checkpoint directions between transition and
publication. Its CLI check blocks imports of models and component builders
while compiling, verifying and resuming the closed case.
`tests/test_core_stateless.py` additionally covers zero-state monitors,
timer-only programs, combinational relay composition and empty-state checkpoints.

The library includes owned FIFOs/rings with explicit initialization and removal,
FIR/quantization, stable minimum-priority arbitration, clocks/timer slots and
bounded repeated transactions. New compositions expand these declarations; they
do not add backend opcodes named after examples. The builders in
`core.primitives` belong to this library layer: they produce ordinary semantic
declarations and need no special handling in the interpreter or compiler.
The two-counter timer/drain
composition in `tests/test_modeling_events.py` checks instance isolation, sampled
ports, current-state repetition, atomic rollback, ordered prefixes and provenance.

The modeling layer deliberately requires explicit transaction/failure order and
entry-sampled ports. It does not infer arbitration, offer unrestricted historical
arrays, or synthesize crossing/handshake policies. Component definitions are flat;
products/enums use named finite fields. Ports currently name a formal periodic
clock domain; standalone unclocked/timer-domain ports are not defined, although
owned timers can activate actions within a component's declared domain and
timer-only core programs need no periodic clock. Origins identify declaration paths and
versions; compilation additionally exports semantic paths to generated source
lines. This is a source map and consistency certificate, not a formal proof.

Common `Session`/`Case` APIs supply resident batches, compact output, continuation,
checkpoint portability and complete replay inputs. Tests compare independent
reference execution with generated CPU/CUDA behavior on bounded programs.
