"""Generic resident scenario batches with exact event-boundary continuation.

The compiler and executor see only a Program. Scenario inputs are data, and
native state stays allocated between steps. One worker owns each scenario.
"""

import threading
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from math import lcm
from time import perf_counter

from .execution import (
    Result,
    check_memory,
    data_offset,
    decode_record,
    prepare,
    state_dict,
    trace_layout,
)
from .lower import MAX, integer, lower, rational
from .reference import Interpreter


@dataclass(frozen=True)
class Scenario:
    drives: tuple = ()
    initial: dict | None = None
    tables: dict | None = None


@dataclass(frozen=True)
class Progress:
    scenario_id: int
    status: str
    time: Fraction | None
    events: int
    microstep: int


class Session:
    def __init__(
        self,
        program,
        scenarios=(Scenario(),),
        *,
        device="cpu",
        mode="release",
        trace_cells=None,
        trace_mode="full",
        memory_limit=256 * 1024**2,
        time_denominator=1,
        cpu_threads=1,
        cpu_schedule="static",
        cpu_chunk=1,
        block_size=128,
        reuse_buffers=True,
        profile=False,
        strategy="scalar",
    ):
        start = perf_counter()
        self.l = lower(program)
        self.metrics = dict(
            lowering_seconds=perf_counter() - start,
            preparation_seconds=0.0,
            source_generation_seconds=0.0,
            compile_load_seconds=0.0,
            allocation_upload_seconds=0.0,
            trace_allocation_enqueue_seconds=0.0,
            native_execution_seconds=0.0,
            synchronization_readback_seconds=0.0,
            decode_seconds=0.0,
            checkpoint_seconds=0.0,
            launches=0,
            trace_allocations=0,
            host_readback_bytes=0,
            device_copy_bytes=0,
            peak_buffer_bytes=0,
            compaction_seconds=0.0,
        )
        start = perf_counter()
        if type(scenarios) is not tuple or not 1 <= len(scenarios) <= 65536:
            raise ValueError("Session requires a bounded tuple of scenarios")
        if any(type(s) is not Scenario for s in scenarios):
            raise ValueError("Expected Scenario declarations")
        self.program, self.scenarios = program, deepcopy(scenarios)
        self.device, self.mode = device, mode
        self.trace_cells, self.trace_mode = trace_cells, trace_mode
        self.memory_limit = memory_limit
        integer(cpu_threads, 1, 256)
        integer(cpu_chunk, 1, 65536)
        if cpu_schedule not in ("static", "dynamic"):
            raise ValueError("CPU schedule must be static or dynamic")
        if block_size not in (1, 32, 64, 128, 256):
            raise ValueError("Block size must be 1, 32, 64, 128 or 256")
        if strategy not in ("scalar", "warp", "block"):
            raise ValueError("Unknown execution strategy")
        if strategy != "scalar" and device in ("cpu", "reference"):
            raise ValueError("Cooperative strategies require CUDA")
        self.cpu_threads, self.cpu_schedule, self.cpu_chunk = (
            cpu_threads,
            cpu_schedule,
            cpu_chunk,
        )
        self.block_size = 32 if strategy == "warp" else block_size
        self.reuse_buffers, self.profile, self.strategy = (
            reuse_buffers,
            profile,
            strategy,
        )
        self.lock = threading.RLock()
        self.until = Fraction(-1)
        self.results = None
        self._dirty = False
        self._event_bound = 0
        self._traces = None
        self._trace_capacity = 0
        self._events = []
        self._ids = self._active_size = None
        self.active_counts = []
        # All lanes share an exact lattice, including every future input time.
        first = [
            prepare(self.l, "0", s.drives, s.initial, 1, s.tables) for s in scenarios
        ]
        integer(time_denominator, 1, MAX)
        self.denominator = lcm(time_denominator, *(p.denominator for p in first))
        self.prepared = [
            prepare(
                self.l,
                "0",
                s.drives,
                s.initial,
                1,
                s.tables,
                time_denominator=self.denominator,
            )
            for s in scenarios
        ]
        self._check_memory(1)
        self.metrics["preparation_seconds"] = perf_counter() - start
        if device == "reference":
            self.interpreters = [
                Interpreter(self.l, p, trace_cells, trace_mode) for p in self.prepared
            ]
        else:
            self._initialize_native()

    def _check_memory(self, budget):
        integer(self.memory_limit, 1024, 64 * 1024**3)
        per_lane = self.memory_limit // len(self.scenarios)
        # Native external storage is padded to the longest input schedule.
        longest = max(self.prepared, key=lambda p: len(p.drives))
        check_memory(
            self.l,
            replace(longest, max_events=budget),
            self.trace_cells,
            per_lane,
            self.trace_mode,
        )

    def _initialize_native(self):
        import numpy as np
        import warp as wp

        from ..native_runtime import compile_source
        from .codegen import Emitter

        target = wp.get_device(self.device)
        if self.strategy != "scalar" and (not target.is_cuda or target.arch < 70):
            raise ValueError(
                "Cooperative strategies require CUDA compute capability 7.0+"
            )
        start = perf_counter()
        emitter = Emitter(
            self.l, self.trace_cells, self.trace_mode, strategy=self.strategy
        )
        source = emitter.generate()
        self.metrics["source_generation_seconds"] = (
            perf_counter() - start - emitter.lowering_seconds
        )
        self.metrics["lowering_seconds"] += emitter.lowering_seconds
        start = perf_counter()
        self.module = compile_source(
            source,
            self.device,
            self.mode,
            openmp=self.cpu_threads > 1 and not target.is_cuda,
        )
        self.metrics["compile_load_seconds"] = perf_counter() - start
        self.cooperative_regions = getattr(emitter, "cooperative_regions", ())
        start = perf_counter()
        self.stream = (
            wp.get_stream(self.device) if wp.get_device(self.device).is_cuda else None
        )
        n, ni = len(self.scenarios), len(self.program.inputs)
        self.data = data_offset(self.program)
        initial = np.asarray(
            [[0] * self.data + p.initial for p in self.prepared], dtype=np.int64
        ).T.copy()
        self.states = wp.array(initial, dtype=wp.int64, device=self.device)
        self.external_size = max(1, max(len(p.drives) for p in self.prepared))
        external = np.zeros((self.external_size, ni + 1, n), dtype=np.int64)
        for i, p in enumerate(self.prepared):
            if p.drives:
                external[: len(p.drives), :, i] = p.drives
        self.external = wp.array(external, dtype=wp.int64, device=self.device)
        self.counts = wp.array(
            np.asarray([len(p.drives) for p in self.prepared], dtype=np.int64),
            dtype=wp.int64,
            device=self.device,
        )
        scales = np.asarray(
            [
                [self.denominator // self.l.denominator, *p.tables]
                for p in self.prepared
            ],
            dtype=np.int64,
        ).T.copy()
        self.scales = wp.array(scales, dtype=wp.int64, device=self.device)
        self._base_bytes = (initial.size + external.size + n + scales.size) * 8
        self.metrics["peak_buffer_bytes"] = self._base_bytes
        self.metrics["allocation_upload_seconds"] = perf_counter() - start

    def _stream_scope(self):
        import warp as wp

        return (
            wp.ScopedStream(self.stream) if self.stream is not None else nullcontext()
        )

    def _validate_advance(self, until, budget):
        horizon = rational(until)
        integer(budget, 1, 100000)
        if self._event_bound > MAX - budget:
            raise ValueError(
                "Cumulative event count could overflow; inspect progress first"
            )
        if horizon < 0 or horizon < self.until:
            raise ValueError("Session horizons must be nonnegative and nondecreasing")
        tick = int(horizon * self.denominator)
        integer(tick, 0, MAX)
        self._check_memory(budget)
        return horizon, tick

    def step(self, *, until, max_events=256):
        """Execute at most max_events additional groups per scenario.

        Results contain cumulative event counts and only this call's records.
        A terminal scenario returns its existing terminal result on later calls.
        Horizon stops are resumable; completion and faults are terminal.
        """
        with self.lock:
            if self.results and not self._dirty:
                self._event_bound = max(r.events for r in self.results)
            horizon, tick = self._validate_advance(until, max_events)
            if self.device == "reference":
                results = tuple(
                    self.results[i]
                    if self.results and self.results[i].status in ("fault", "complete")
                    else engine.run(tick, max_events)
                    for i, engine in enumerate(self.interpreters)
                )
            else:
                results = self._step_native(tick, max_events)
            self.until, self.results = horizon, results
            self._event_bound = max(r.events for r in results)
            return results

    def _step_native(self, tick, budget):
        self._advance_native(tick, budget)
        return self.inspect()

    def _reserve_trace(self, count):
        import warp as wp

        if self.reuse_buffers and self._trace_capacity >= count:
            return
        start = perf_counter()
        _, indices = trace_layout(self.l, self.trace_cells)
        size = count * (18 + len(indices)) * len(self.scenarios) * 8
        old_bytes = 0 if self._traces is None else self._traces.size * 8
        peak = self._base_bytes + size + old_bytes
        if peak > self.memory_limit:
            raise ValueError("Trace allocation exceeds packed-buffer memory budget")
        traces = wp.empty(
            (count, 18 + len(indices), len(self.scenarios)),
            dtype=wp.int64,
            device=self.device,
        )
        # Uninspected fault records remain owned by terminal lanes across advances.
        if self.trace_mode == "summary" and self._traces is not None:
            wp.copy(traces, self._traces)
            self.metrics["device_copy_bytes"] += old_bytes
        self._traces, self._trace_capacity = traces, count
        self.metrics["trace_allocations"] += 1
        self.metrics["peak_buffer_bytes"] = max(self.metrics["peak_buffer_bytes"], peak)
        self.metrics["trace_allocation_enqueue_seconds"] += perf_counter() - start

    def _advance_native(self, tick, budget, active_count=None):
        import warp as wp

        from ..native_runtime import INT, LONG, POINTER

        n = len(self.scenarios)
        count = 1 if self.trace_mode == "summary" else budget * self.l.records_per_group
        with self._stream_scope():
            self._reserve_trace(count)
        args = [
            POINTER(self.states.ptr),
            POINTER(self.external.ptr),
            POINTER(self.counts.ptr),
            POINTER(self.scales.ptr),
            INT(n),
            INT(self.external_size),
            LONG(tick),
            INT(budget),
            POINTER(self._traces.ptr),
            INT(count),
        ]
        size = n if active_count is None else active_count
        types = [POINTER] * 4 + [INT, INT, LONG, INT, POINTER, INT]
        function = self.module.advance
        if self.stream is None and (self.cpu_threads > 1 or active_count is not None):
            function = self.module.function(
                "advance_parallel", types + [POINTER, INT, INT, INT, INT]
            )
            args += [
                POINTER(0 if active_count is None else self._ids.ptr),
                INT(size),
                INT(self.cpu_threads),
                INT(self.cpu_schedule == "dynamic"),
                INT(self.cpu_chunk),
            ]
        elif active_count is not None:
            function = self.module.function("advance_indexed")
            args += [POINTER(self._ids.ptr), INT(size)]
        launch_size = size * self.block_size if self.strategy != "scalar" else size
        start = perf_counter()
        with self._stream_scope():
            if self.profile and self.stream is not None:
                begin, end = (
                    wp.Event(self.device, enable_timing=True),
                    wp.Event(self.device, enable_timing=True),
                )
                self.stream.record_event(begin)
            self.module.launch(function, args, launch_size, self.block_size)
            if self.profile and self.stream is not None:
                self.stream.record_event(end)
                self._events.append((begin, end))
        if self.profile and self.stream is None:
            self.metrics["native_execution_seconds"] += perf_counter() - start
        self.metrics["launches"] += 1
        self._event_bound += budget
        self._dirty = True

    def _finish_events(self):
        if self._events:
            import warp as wp

            self.metrics["native_execution_seconds"] += sum(
                wp.get_event_elapsed_time(a, b) * 0.001 for a, b in self._events
            )
            self._events.clear()

    def advance(self, *, until, max_events=256):
        """Enqueue a summary advance; inspect observes the latest completed state.

        No historical lazy result is returned. Detailed traces use eager step/run.
        """
        with self.lock:
            if self.device == "reference" or self.trace_mode != "summary":
                raise ValueError("Resident advance requires a native summary session")
            horizon, tick = self._validate_advance(until, max_events)
            self._advance_native(tick, max_events)
            self.until = horizon

    def status(self):
        """Read compact progress in original scenario order without model-state decoding."""
        with self.lock:
            if self.device == "reference":
                if self.results is None:
                    return tuple(
                        Progress(i, "ready", None, 0, 0)
                        for i in range(len(self.scenarios))
                    )
                return tuple(
                    Progress(i, r.status, r.time, r.events, r.microstep)
                    for i, r in enumerate(self.results)
                )
            start = perf_counter()
            with self._stream_scope():
                rows = self.states[:12].numpy()
            self._finish_events()
            self.metrics["synchronization_readback_seconds"] += perf_counter() - start
            self.metrics["host_readback_bytes"] += rows.size * 8
            self._event_bound = int(max(rows[2]))
            return tuple(
                Progress(
                    i,
                    {0: "ready", 1: "horizon", 2: "budget", 3: "fault", 4: "complete"}[
                        int(row[0])
                    ],
                    Fraction(int(row[1]), self.denominator)
                    if row[10] and row[1] >= 0
                    else None,
                    int(row[2]),
                    int(row[9]),
                )
                for i, row in enumerate(rows.T)
            )

    def inspect(self):
        """Materialize complete current results; returned objects own their data."""
        with self.lock:
            if not self._dirty:
                if self.results is None:
                    raise ValueError("Advance the session before inspection")
                return self.results
            self.results = self._inspect_native()
            self._dirty = False
            self._event_bound = max(r.events for r in self.results)
            return self.results

    def _inspect_native(self):
        from .native import decode_calendar

        n = len(self.scenarios)
        start = perf_counter()
        with self._stream_scope():
            states = self.states.numpy()
        self._finish_events()
        lengths = [int(states[3, i]) for i in range(n)]
        if any(c < 0 or c > self._trace_capacity for c in lengths):
            raise RuntimeError("Native trace bound violated")
        with self._stream_scope():
            packed = self._traces[: max(lengths)].numpy() if max(lengths) else None
        self.metrics["synchronization_readback_seconds"] += perf_counter() - start
        self.metrics["host_readback_bytes"] += states.size * 8 + (
            packed.size * 8 if packed is not None else 0
        )
        start = perf_counter()
        results = []
        for i in range(n):
            if self.results and self.results[i].status in ("fault", "complete"):
                results.append(self.results[i])
                continue
            row = states[:, i]
            records = tuple(
                decode_record(
                    self.l, self.denominator, packed[j, :, i], self.trace_cells
                )
                for j in range(lengths[i])
            )
            results.append(
                Result(
                    self.l.digest,
                    Fraction(1, self.denominator),
                    {1: "horizon", 2: "budget", 3: "fault", 4: "complete"}[int(row[0])],
                    Fraction(int(row[1]), self.denominator) if row[1] >= 0 else None,
                    int(row[2]),
                    state_dict(self.l, row[self.data :]),
                    records,
                    int(row[9]),
                    decode_calendar(self.l, row),
                )
            )
        self.metrics["decode_seconds"] += perf_counter() - start
        return tuple(results)

    def _compact(self):
        import warp as wp
        from ..native_runtime import INT, POINTER

        start = perf_counter()
        n = len(self.scenarios)
        with self._stream_scope():
            if self._ids is None:
                extra = (n + 1) * 4
                if self._base_bytes + extra + self._traces.size * 8 > self.memory_limit:
                    raise ValueError("Compaction exceeds packed-buffer memory budget")
                self._ids = wp.empty(n, dtype=wp.int32, device=self.device)
                self._active_size = wp.zeros(1, dtype=wp.int32, device=self.device)
                self._base_bytes += extra
                self.metrics["peak_buffer_bytes"] = max(
                    self.metrics["peak_buffer_bytes"],
                    self._base_bytes + self._traces.size * 8,
                )
            else:
                self._active_size.zero_()
            function = self.module.function(
                "compact_active", [POINTER, INT, POINTER, POINTER]
            )
            self.module.launch(
                function,
                [
                    POINTER(self.states.ptr),
                    INT(n),
                    POINTER(self._ids.ptr),
                    POINTER(self._active_size.ptr),
                ],
                n,
            )
            count = int(self._active_size.numpy()[0])
        self._finish_events()
        self.metrics["host_readback_bytes"] += 4
        self.metrics["compaction_seconds"] += perf_counter() - start
        self.active_counts.append(count)
        return count

    def campaign(
        self,
        *,
        until,
        max_events=256,
        quantum=1000,
        compact=False,
        poll_interval=1,
        filter_active=True,
    ):
        """Run summary work with scheduling yields independent of semantic budgets."""
        with self.lock:
            if self.device == "reference" or self.trace_mode != "summary":
                raise ValueError("Resident campaign requires a native summary session")
            integer(max_events, 1, MAX)
            integer(quantum, 1, 100000)
            integer(poll_interval, 1, 100000)
            remaining, iteration, active_count = max_events, 0, None
            while remaining:
                budget = min(quantum, remaining)
                horizon, tick = self._validate_advance(until, budget)
                self._advance_native(tick, budget, active_count)
                self.until = horizon
                remaining -= budget
                iteration += 1
                if iteration % poll_interval == 0 and remaining:
                    if compact:
                        count = self._compact()
                        if not count:
                            break
                        active_count = count if filter_active else None
                    elif not any(p.status == "budget" for p in self.status()):
                        break
            return self.inspect()

    def run(self, *, until, max_events=256):
        """Run a bounded campaign using chunks of at most 100,000 event groups.

        Full output is budgeted for the entire campaign; summary output retains
        only the first fault. A budget outcome remains resumable with step/run.
        """
        with self.lock:
            integer(max_events, 1, MAX)
            self._check_memory(max_events)
            records = [[] for _ in self.scenarios]
            remaining = max_events
            previous = None
            while remaining:
                quantum = min(100000, remaining)
                results = self.step(until=until, max_events=quantum)
                for i, r in enumerate(results):
                    if previous is None or previous[i].status not in (
                        "fault",
                        "complete",
                    ):
                        records[i].extend(r.records)
                remaining -= quantum
                if not any(r.status == "budget" for r in results):
                    break
                previous = results
            self.results = tuple(
                replace(r, records=tuple(records[i])) for i, r in enumerate(results)
            )
            return self.results

    def checkpoint(self, path):
        from .serialization import save_checkpoint

        with self.lock:
            start = perf_counter()
            if self._dirty:
                self.inspect()
            save_checkpoint(self, path)
            self.metrics["checkpoint_seconds"] += perf_counter() - start

    @classmethod
    def restore(cls, path, **options):
        from .serialization import restore_checkpoint

        return restore_checkpoint(path, **options)
