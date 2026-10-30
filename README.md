# WarpSim

- Reusable finite-state modeling with typed state, exact discrete time, transactions, timers and component composition.
- The same model runs on an independent Python interpreter, native CPU or CUDA. CPU execution needs Python 3.12+ and a C++17 compiler; CUDA also needs a compatible NVIDIA driver.
- Install: `uv sync --locked --extra test`.
- **Example:** [queued FIR pipeline](examples/queued_fir.py). Run it with `uv run python examples/queued_fir.py --device cpu --output /tmp/queued-fir.json`; it checks native execution against the interpreter.
- Another composition example: [traffic signals](examples/traffic_signals.py).
- Tests: `uv run python -m pytest -q`.
- API and semantics: [CORE.md](CORE.md), [MODELING.md](MODELING.md), [EXECUTION.md](EXECUTION.md).
- Models declare application behavior; the compiler and runtime operate on the shared representation. Execution is bounded, and passing finite runs does not establish universal safety.

A counter that increments every half second, including at time zero:

```python
from warpsim.core import (
    Bits, Cell, Clock, Program, Region, Store,
    const, due, op, read, run_native,
)

u8 = Bits(8)
counter = Program(
    "counter",
    (Cell("count", u8),),
    (Clock("tick", "1/2"),),
    (Region("increment", (
        Store("count", op("add", read("count", u8), const(1, u8))),
    ), due("tick")),),
)
result = run_native(counter, device="cpu", until="2", max_events=8)
print(result.state["count"])  # (5,)
```
