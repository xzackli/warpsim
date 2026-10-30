# Execution API

`Session(program, scenarios, device=...)` owns a batch of scenarios sharing one
program and exact time lattice. Each `Scenario` supplies drives, initial state,
and immutable runtime tables. Supported devices are `reference`, `cpu`, and CUDA
device names such as `cuda:0`.

```python
from warpsim.core import Session, Scenario
from warpsim.models.fifo_pipeline import fifo_pipeline

session = Session(
    fifo_pipeline().program, (Scenario(), Scenario()),
    device="cpu", trace_mode="summary", cpu_threads=2,
)
session.advance(until="12", max_events=4)
progress = session.status()
session.checkpoint("/tmp/pipeline.checkpoint.json")
restored = Session.restore("/tmp/pipeline.checkpoint.json", device="reference")
results = restored.run(until="12", max_events=100)
```

## Bounded execution and inspection

- `step` returns full results for at most 100,000 additional event groups per
  scenario. `run` supports a larger finite budget by executing bounded chunks.
- `advance` executes a summary-mode quantum while retaining native state.
  `status` reads compact progress; `inspect` explicitly collects complete results.
- `campaign(until=..., max_events=..., quantum=..., compact=..., poll_interval=...)`
  orchestrates summary-mode execution and final inspection. Optional compaction
  filters completed scenarios while retaining their original scenario IDs.
- Results distinguish `complete`, `fault`, `horizon`, `budget`, and other stop
  conditions. Horizon and budget stops can continue with a nondecreasing horizon.
  Event counts are cumulative. Event budgets limit additional work for that call.
- `trace_mode` selects full records, observations/faults, or summaries. Choosing
  `trace_cells` limits diagnostic snapshots without changing final state.

Checkpoints carry logical state, calendar, program, and stimuli. They can be
restored on another supported backend. A checksum detects accidental changes;
validation checks structural consistency, not reachability of arbitrary edited
states. Successful trace history is not reconstructed by a checkpoint.

## Execution strategies

Native CPU sessions use `cpu_threads`, `cpu_schedule` (`static` or `dynamic`), and
`cpu_chunk`. Multi-threaded execution uses OpenMP. Scalar CUDA execution assigns
one scenario to a thread. `strategy="warp"` or `"block"` requests conservative
cooperative lowering for supported independent indexed operations; unsupported
programs are rejected. A strategy is an execution choice, not a semantic policy.

`reuse_buffers=True` retains scratch allocations. `profile=True` enables additional
timing instrumentation in `session.metrics`; instrumentation can affect elapsed
time. Inspect generated source and the returned diagnostic locations when needed:

```sh
uv run vlsi-sim core compile /tmp/pipeline.json --output /tmp/pipeline.cpp
uv run vlsi-sim core run /tmp/pipeline.json --device cpu --resident \
  --cpu-threads 2 --quantum 4 --output /tmp/pipeline.run.json
```

`scripts/benchmark_core.py --help` describes neutral benchmark fixtures and timing
controls. It verifies results before measuring and reports setup, compilation,
execution, and collection separately. Compare equivalent workloads and output
collection when assessing a backend. No strategy is assumed universally faster.
