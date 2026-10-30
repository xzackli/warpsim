"""Reproducible semantic-core campaign experiments, not a model-specific runner.

Example (run from the repository's installed environment):
  python scripts/benchmark_core.py --fixtures map skew --batches 1 32 \
      --sizes 64 --quanta 8 --events 32 --repeats 3 --output /tmp/core.json

Timings compare complete Results against an independent interpreter. Warm
measurements are unprofiled; an optional separate sample collects device events.
The cold option launches a new process with empty native/Warp cache directories.
"""

import argparse
from collections import Counter
from dataclasses import replace
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

from warpsim.campaign_fixtures import FIXTURES, build_case
from warpsim.core import Case, Session
from warpsim.core.serialization import canonical, encode


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "reference",
    "cpu-serial",
    "cpu-static",
    "cpu-dynamic",
    "cpu-resident-serial",
    "cpu-resident-static",
    "cpu-resident-dynamic",
    "cpu-count-only-serial",
    "cpu-count-only-static",
    "cpu-count-only-dynamic",
    "cpu-compact-serial",
    "cpu-compact-static",
    "cpu-compact-dynamic",
    "cuda-eager",
    "cuda-eager-reuse",
    "cuda-resident",
    "cuda-compact",
    "cuda-count-only",
    "cuda-map-serial",
    "cuda-warp",
    "cuda-block",
)
TIMING_KEYS = (
    "setup_seconds",
    "execution_and_materialization_seconds",
    "campaign_result_seconds",
    "intermediate_checkpoint_seconds",
    "checkpoint_seconds",
    "restore_seconds",
    "campaign_with_checkpoint_restore_seconds",
)


def digest(value):
    return hashlib.sha256(canonical(encode(value)).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def command_output(command):
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=15
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def cpu_topology():
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    physical = set()
    for cpu in affinity:
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            physical.add(
                (
                    (directory / "physical_package_id").read_text().strip(),
                    (directory / "core_id").read_text().strip(),
                )
            )
        except OSError:
            pass
    return {
        "logical_cpus": os.cpu_count(),
        "allowed_cpus": affinity,
        "allowed_physical_cores": len(physical) or None,
    }


def rss_bytes():
    # Linux ru_maxrss is KiB, macOS bytes. This is a process-lifetime high-water
    # mark, never mislabeled as a particular sample's incremental footprint.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )


def source_hashes():
    paths = sorted(
        p
        for p in (ROOT / "src" / "warpsim").rglob("*")
        if p.suffix in (".py", ".cpp", ".h")
    )
    paths += [Path(__file__).resolve(), ROOT / "pyproject.toml", ROOT / "uv.lock"]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def environment():
    packages = {}
    for name in ("warp-lang", "numpy", "warpsim-core"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": cpu_topology(),
        "packages": packages,
        "compiler": command_output(["g++", "--version"]),
        "gpu_query": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,pstate,clocks.sm,clocks.mem,power.limit",
                "--format=csv",
            ]
        ),
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--porcelain"]),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "OMP_DYNAMIC",
                "OMP_THREAD_LIMIT",
                "CUDA_VISIBLE_DEVICES",
                "XDG_CACHE_HOME",
            )
        },
        "rss_scope": "process-lifetime high-water mark; warm samples share a process",
    }


def initialize_runtime(*, cuda_contexts=True):
    start = perf_counter()
    import warp as wp

    wp.config.log_level = wp.LOG_WARNING
    if os.environ.get("WARPSIM_BENCHMARK_COLD_CACHE"):
        wp.config.kernel_cache_dir = str(
            Path(os.environ["WARPSIM_BENCHMARK_COLD_CACHE"]) / "warp"
        )
    wp.init()
    devices = []
    for device in wp.get_devices():
        if device.is_cuda and cuda_contexts:
            # Force lazy context/stream creation into the separately reported
            # initialization cost, including in fresh cold workers.
            wp.synchronize_stream(wp.get_stream(device))
        devices.append(
            {
                "alias": device.alias,
                "name": device.name,
                "arch": device.arch,
                "is_cuda": device.is_cuda,
            }
        )
    return {
        "seconds": perf_counter() - start,
        "devices": devices,
        "cuda_available": wp.is_cuda_available(),
        "cuda_contexts_initialized": cuda_contexts,
    }


def variant_options(name, config, *, profile=False):
    options = dict(
        device="reference"
        if name == "reference"
        else "cuda:0"
        if name.startswith("cuda-")
        else "cpu",
        mode=config["mode"],
        trace_mode="summary",
        memory_limit=config["memory_limit"],
        cpu_threads=config["cpu_threads"]
        if name.startswith("cpu-") and name.endswith(("-static", "-dynamic"))
        else 1,
        cpu_schedule="dynamic"
        if name.startswith("cpu-") and name.endswith("-dynamic")
        else "static",
        cpu_chunk=config["cpu_chunk"],
        block_size=1 if name == "cuda-map-serial" else config["block_size"],
        reuse_buffers=name != "cuda-eager",
        profile=profile,
        strategy="block"
        if name == "cuda-map-serial"
        else name.removeprefix("cuda-")
        if name in ("cuda-warp", "cuda-block", "cuda-map-serial")
        else "scalar",
    )
    return options


def execute(session, case, config, name, checkpoint):
    interval = config.get("checkpoint_interval", 0)
    count_only = name == "cuda-count-only" or name.startswith("cpu-count-only-")
    compact = count_only or name == "cuda-compact" or name.startswith("cpu-compact-")
    resident = name.startswith(
        ("cpu-resident-", "cpu-count-only-", "cpu-compact-")
    ) or name in (
        "cuda-resident",
        "cuda-compact",
        "cuda-count-only",
        "cuda-warp",
        "cuda-block",
        "cuda-map-serial",
    )
    remaining = case.max_events
    iteration, intermediate_checkpoints = 0, 0
    while remaining:
        if resident:
            count = (
                min(remaining, interval * config["quantum"]) if interval else remaining
            )
            result = session.campaign(
                until=case.until,
                max_events=count,
                quantum=config["quantum"],
                compact=compact,
                filter_active=not count_only,
                poll_interval=config["poll_interval"],
            )
        else:
            count = min(remaining, config["quantum"])
            result = session.step(until=case.until, max_events=count)
        iteration += 1
        remaining -= count
        if not any(r.status == "budget" for r in result):
            break
        if remaining and interval and (resident or iteration % interval == 0):
            session.checkpoint(checkpoint)
            intermediate_checkpoints += 1
    return result, intermediate_checkpoints


def measure(case, name, config, *, profile=False):
    options = variant_options(name, config, profile=profile)
    with tempfile.TemporaryDirectory(prefix="warpsim-core-benchmark-") as scratch:
        checkpoint = Path(scratch) / "checkpoint.json"
        start = perf_counter()
        session = Session(
            case.program,
            case.scenarios,
            trace_cells=case.trace_cells,
            time_denominator=case.time_denominator,
            **options,
        )
        if getattr(session, "stream", None) is not None:
            import warp as wp

            wp.synchronize_stream(session.stream)
        setup_end = perf_counter()
        results, intermediate_checkpoints = execute(
            session, case, config, name, checkpoint
        )
        execution_end = perf_counter()
        intermediate_checkpoint_seconds = session.metrics["checkpoint_seconds"]
        session.checkpoint(checkpoint)
        checkpoint_end = perf_counter()
        restored = Session.restore(
            checkpoint, **{k: v for k, v in options.items() if k != "trace_mode"}
        )
        restored_results = restored.inspect()
        if getattr(restored, "stream", None) is not None:
            import warp as wp

            wp.synchronize_stream(restored.stream)
        restore_end = perf_counter()
        # Validation and hashing occur outside all measured critical paths.
        if restored_results != results:
            raise AssertionError(
                "Checkpoint restore changed state, calendar or diagnostics"
            )
        module = getattr(session, "module", None)
        module_info = (
            None
            if module is None
            else {
                "compile_seconds": module.compile_seconds,
                "load_seconds": module.load_seconds,
                "cache_hit_at_module_construction": module.cache_hit,
                "source_path": str(module.source_path),
                "source_bytes": module.source_path.stat().st_size,
                "source_sha256": hashlib.sha256(
                    module.source_path.read_bytes()
                ).hexdigest(),
                "binary_bytes": module.binary_path.stat().st_size,
                "binary_sha256": hashlib.sha256(
                    module.binary_path.read_bytes()
                ).hexdigest(),
                "kernel_properties": module.kernel_properties,
                "kernel_properties_symbol": "advance (not advance_indexed)",
                "scope": "original module construction; in-process module cache may reuse this object",
            }
        )
        return {
            "setup_seconds": setup_end - start,
            "execution_and_materialization_seconds": execution_end - setup_end,
            "campaign_result_seconds": execution_end - start,
            "intermediate_checkpoint_count": intermediate_checkpoints,
            "intermediate_checkpoint_seconds": intermediate_checkpoint_seconds,
            "checkpoint_seconds": checkpoint_end - execution_end,
            "restore_seconds": restore_end - checkpoint_end,
            "campaign_with_checkpoint_restore_seconds": restore_end - start,
            "metrics": dict(session.metrics),
            "active_count_history": session.active_counts,
            "restore_metrics": dict(restored.metrics),
            "module": module_info,
            "cooperative_regions": encode(getattr(session, "cooperative_regions", ())),
            "effective_options": {
                "device": session.device,
                "cpu_threads": session.cpu_threads,
                "cpu_schedule": session.cpu_schedule,
                "cpu_chunk": session.cpu_chunk,
                "block_size": session.block_size,
                "strategy": session.strategy,
                "reuse_buffers": session.reuse_buffers,
            },
            "checkpoint_bytes": checkpoint.stat().st_size,
            "process_peak_rss_bytes": rss_bytes(),
            "result_sha256": digest(results),
            "outcomes": dict(Counter(r.status for r in results)),
            "events": sum(r.events for r in results),
            "instrumented": profile,
        }, results


def reference_work(case):
    """Evaluate every unique full Scenario, then scale by exact multiplicities.

    The model, horizon, lattice and budget are shared. Only byte-equivalent
    Scenario declarations are deduplicated. Full trace records use the Case's
    explicitly selected trace cells, while final state comparison covers all
    cells. The interpreter is never replaced by native execution for this step.
    """
    start = perf_counter()
    unique, multiplicities, order = {}, Counter(), []
    for scenario in case.scenarios:
        key = digest(scenario)
        unique.setdefault(key, scenario)
        multiplicities[key] += 1
        order.append(key)
    expected, prototypes = {}, []
    counts = Counter()
    lattice = Session(
        case.program,
        tuple(unique.values()),
        device="reference",
        trace_mode="summary",
        time_denominator=case.time_denominator,
    ).denominator
    for key, scenario in unique.items():
        session = Session(
            case.program,
            (scenario,),
            device="reference",
            trace_mode="full",
            trace_cells=case.trace_cells,
            time_denominator=lattice,
        )
        remaining = case.max_events
        kinds, timer_occurrences = Counter(), Counter()
        failed, trace_records = 0, 0
        # Bound temporary full-trace memory independently of campaign duration.
        while remaining:
            quantum = min(1024, remaining)
            result = session.step(until=case.until, max_events=quantum)[0]
            remaining -= quantum
            kinds.update(r.kind for r in result.records)
            failed += sum(
                r.kind == "fault" and r.region is not None for r in result.records
            )
            trace_records += len(result.records)
            if case.program.timers:
                for record in result.records:
                    if record.observation == "sources":
                        for i, timer in enumerate(case.program.timers):
                            timer_occurrences[timer.name] += record.values[
                                i + len(case.program.clocks)
                            ]
            if result.status != "budget":
                break
        expected[key] = replace(
            result, records=tuple(r for r in result.records if r.kind == "fault")
        )
        clock_occurrences = {}
        for clock in case.program.clocks:
            clock_occurrences[clock.name] = (
                0
                if result.time is None
                else max(
                    0,
                    int((result.time - Fraction(clock.phase)) // Fraction(clock.period))
                    + 1,
                )
            )
        # This fixture declares an ordinary observation of all due flags. Timer
        # generations alone cannot distinguish dispatched from cancelled work.
        item = {
            "scenario_sha256": key,
            "multiplicity": multiplicities[key],
            "event_groups": result.events,
            "transactions_committed": kinds["commit"],
            "transactions_failed": failed,
            "guards_disabled": kinds["disabled"],
            "observations": kinds["observe"],
            "periodic_occurrences": clock_occurrences,
            "timer_occurrences": dict(timer_occurrences),
            "packed_full_trace_bytes": trace_records
            * (18 + sum(len(result.state[name]) for name in case.trace_cells))
            * 8,
            "status": result.status,
            "final_tag": [str(result.time), result.microstep],
            "fault": encode(result.fault),
            "result_sha256": digest(expected[key]),
        }
        item["transactions_attempted"] = (
            item["transactions_committed"] + item["transactions_failed"]
        )
        prototypes.append(item)
        for field in (
            "event_groups",
            "transactions_attempted",
            "transactions_committed",
            "transactions_failed",
            "guards_disabled",
            "observations",
            "packed_full_trace_bytes",
        ):
            counts[field] += item[field] * item["multiplicity"]
        counts["periodic_occurrences"] += (
            sum(clock_occurrences.values()) * item["multiplicity"]
        )
        counts["timer_occurrences"] += (
            sum(timer_occurrences.values()) * item["multiplicity"]
        )
        counts[f"outcome_{result.status}"] += item["multiplicity"]
    results = tuple(expected[key] for key in order)
    return results, {
        "method": "full-trace independent interpreter in bounded 1024-group chunks on every distinct serialized Scenario; exact multiplicities scale work only, never timings",
        "seconds": perf_counter() - start,
        "unique_scenarios": len(unique),
        "prototypes": prototypes,
        "totals": dict(counts),
        "result_sha256": digest(results),
        "transaction_definition": "enabled commits plus region faults; disabled guards reported separately",
        "trace_bytes_definition": "logical packed full-trace records with selected trace cells, not allocation capacity or summary-run transfer bytes",
    }


def distribution(values, seed=0):
    values = list(values)
    rng = random.Random(seed)
    # Descriptive percentile-bootstrap CI for the mean, not a precision claim;
    # small samples and shared machine noise remain explicitly visible.
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000)
    )
    return {
        "n": len(values),
        "samples": values,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        "mean_bootstrap_95_percent_interval": [means[49], means[1949]],
    }


def summarize(samples, work):
    summaries = {
        key: distribution(sample[key] for sample in samples) for key in TIMING_KEYS
    }
    events = work["totals"]["event_groups"]
    outcomes = sum(v for k, v in work["totals"].items() if k.startswith("outcome_"))
    summaries["equivalent_event_groups_per_execution_second"] = distribution(
        events / sample["execution_and_materialization_seconds"] for sample in samples
    )
    summaries["outcomes_per_campaign_result_second"] = distribution(
        outcomes / sample["campaign_result_seconds"] for sample in samples
    )
    summaries["outcomes_by_status_per_campaign_result_second"] = {
        k.removeprefix("outcome_"): distribution(
            v / sample["campaign_result_seconds"] for sample in samples
        )
        for k, v in work["totals"].items()
        if k.startswith("outcome_")
    }
    return summaries


def classify_error(error):
    message = str(error)
    if isinstance(error, subprocess.TimeoutExpired):
        return {
            "status": "unavailable_compilation_budget",
            "error": message,
            "exception": type(error).__name__,
        }
    if (
        isinstance(error, MemoryError)
        or isinstance(error, ValueError)
        and "memory budget" in message.lower()
    ):
        return {
            "status": "unavailable_memory",
            "error": message,
            "exception": type(error).__name__,
        }
    return {"status": "error", "error": message, "exception": type(error).__name__}


def cold_sample(case_path, name, config, expected_hash, timeout, *, isolated=True):
    with tempfile.TemporaryDirectory(prefix="warpsim-core-cold-") as scratch:
        directory = Path(scratch)
        request = directory / "request.json"
        output = directory / "result.json"
        write_json(request, {"case": str(case_path), "variant": name, "config": config})
        env = (
            dict(
                os.environ,
                XDG_CACHE_HOME=str(directory / "cache"),
                WARPSIM_BENCHMARK_COLD_CACHE=str(directory / "cache"),
            )
            if isolated
            else dict(os.environ)
        )
        started = perf_counter()
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(request),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            return {
                "status": "timeout",
                "timeout_seconds": timeout,
                "process_wall_seconds": perf_counter() - started,
                "error": str(error),
            }
        elapsed = perf_counter() - started
        if not output.exists():
            return {
                "status": "error",
                "returncode": completed.returncode,
                "stderr": completed.stderr[-8000:],
                "stdout": completed.stdout[-8000:],
                "process_wall_seconds": elapsed,
            }
        result = json.loads(output.read_text())
        result.update(
            process_wall_seconds=elapsed,
            cache_scope="fresh process, empty native and Warp filesystem caches; no claim of cold GPU hardware caches"
            if isolated
            else "bounded fresh process using shared compilation caches; qualification only",
        )
        if (
            result.get("status") == "ok"
            and result["sample"]["result_sha256"] != expected_hash
        ):
            result.update(
                status="error", error="Cold result differs from independent reference"
            )
        return result


def worker(args):
    request = json.loads(args.worker.read_text())
    try:
        if request["config"].get("cpu_affinity") and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(request["config"]["cpu_affinity"]))
        entry_affinity = cpu_topology()
        started = perf_counter()
        case = Case.load(request["case"])
        load_seconds = perf_counter() - started
        runtime = (
            initialize_runtime(cuda_contexts=request["variant"].startswith("cuda-"))
            if request["variant"] != "reference"
            else None
        )
        sample, _ = measure(case, request["variant"], request["config"])
        value = {
            "status": "ok",
            "case_load_seconds": load_seconds,
            "runtime_initialization": runtime,
            "cpu_affinity_at_entry": entry_affinity,
            "cpu_affinity_after_execution": cpu_topology(),
            "sample": sample,
        }
    except Exception as error:
        value = classify_error(error)
    write_json(args.output, value)
    return int(value["status"] == "error")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--fixtures", nargs="+", choices=FIXTURES, default=["control", "map", "skew"]
    )
    p.add_argument("--batches", nargs="+", type=int, default=[32])
    p.add_argument("--sizes", nargs="+", type=int, default=[64])
    p.add_argument("--quanta", nargs="+", type=int, default=[16])
    p.add_argument("--events", type=int, default=64)
    p.add_argument(
        "--max-events",
        type=int,
        help="Override the Case's semantic event-group budget independently of event targets and scheduling quanta",
    )
    p.add_argument(
        "--distributions",
        nargs="+",
        choices=("identical", "diverse"),
        default=["diverse"],
    )
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument(
        "--ordering",
        choices=("interleaved", "grouped"),
        default="interleaved",
        help="Group byte-identical scenarios to vary assignment skew with a fixed multiset",
    )
    p.add_argument("--cpu-threads", type=int, default=None)
    p.add_argument("--cpu-chunk", type=int, default=1)
    p.add_argument("--cpus", help="Optional explicit Linux affinity list, e.g. 0,2,4,6")
    p.add_argument("--block-size", type=int, choices=(1, 32, 64, 128, 256), default=128)
    p.add_argument("--poll-interval", type=int, default=1)
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Checkpoint every N scheduling quanta while work remains; 0 writes only the final checkpoint",
    )
    p.add_argument("--memory-mib", type=int, default=256)
    p.add_argument("--mode", choices=("release", "debug"), default="release")
    p.add_argument(
        "--no-profile",
        action="store_true",
        help="Omit the separate instrumented sample",
    )
    p.add_argument(
        "--cold",
        action="store_true",
        help="One isolated-cache subprocess per selected configuration and variant",
    )
    p.add_argument("--cold-timeout", type=int, default=600)
    p.add_argument(
        "--preflight-timeout",
        type=int,
        default=0,
        help="Bound each native warmup in a shared-cache worker before loading it (0 disables)",
    )
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    if args.worker:
        return worker(args)
    if args.repeats < 3:
        raise ValueError("At least three warm repetitions are required")
    if any(not 1 <= quantum <= 100000 for quantum in args.quanta):
        raise ValueError("Scheduling quanta must lie in [1, 100000]")
    if (
        args.poll_interval < 1
        or args.memory_mib < 1
        or args.cold_timeout < 1
        or args.preflight_timeout < 0
    ):
        raise ValueError("Poll interval, memory and cold timeout must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("Checkpoint interval must be nonnegative")
    if args.max_events is not None and not 1 <= args.max_events < (1 << 62):
        raise ValueError("Semantic max-events must lie in [1, 2**62-1]")
    if args.cpus:
        os.sched_setaffinity(0, {int(x) for x in args.cpus.split(",")})
    topology = cpu_topology()
    cpu_threads = args.cpu_threads or min(
        8, topology["allowed_physical_cores"] or len(topology["allowed_cpus"])
    )
    if not 1 <= cpu_threads <= 256:
        raise ValueError("CPU thread count must lie in [1, 256]")
    hashes = source_hashes()
    report = {
        "schema": "warpsim.core.campaign-benchmark/v1",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "environment": environment(),
        "source_hashes": hashes,
        "methodology": {
            "warm": "One untimed-to-comparison warmup per variant, followed by randomized unprofiled repetitions; each sample constructs fresh state and returns complete summary Results",
            "correctness": "Exact complete Result equality, including every state cell, calendar, lattice, fault operands and semantic location; independent full-trace interpreter supplies expected outcomes",
            "critical_paths": "campaign_result is Session construction through final Results; setup waits for initial CUDA uploads. campaign_with_checkpoint_restore extends through checkpoint serialization, restored Results and readiness of restored device state. Model construction, reference qualification, context initialization and validation are separate",
            "stages": "Session counters are inclusive instrumentation, not additive wall-time partitions. Native execution may overlap host work or synchronization; checkpoint includes snapshot when needed. Do not sum overlapping metrics",
            "profile": "Instrumented device-event sample is separate from all performance distributions",
            "confidence": "95% percentile-bootstrap intervals describe resampled means; three repetitions do not establish narrow uncertainty or immunity to host contention",
            "cuda_eager": "cuda-eager allocates each step; cuda-eager-reuse only changes reuse; cuda-resident additionally uses compact progress and final inspection; cuda-compact additionally schedules active IDs",
            "cpu": "Serial, static OpenMP and dynamic OpenMP share emitted semantics; cpu-resident-* also match cuda-resident polling and final-inspection cadence. Eager cpu-* inspect every quantum. Thread counts and process affinity are reported",
            "checkpoint_cadence": "Intermediate checkpoints overwrite the same portable file every requested number of scheduling quanta while work and budget remain. Their inclusive cost is inside execution wall; checkpoint_seconds is the final checkpoint alone, while Session checkpoint_seconds includes all checkpoints. Native campaigns materialize at checkpoint boundaries and continue the same session",
            "checkpoint_storage": "Temporary-directory filesystem, no fsync: serialization and reload costs, not durable storage latency",
            "preflight": "Optional bounded shared-cache worker validates a native case before parent warmup; timeouts are explicit unavailable cases, excluded from timing distributions",
            "compaction_control": "cuda-count-only and cpu-count-only-* perform native compaction/count readback but continue launching all scenario slots, isolating indexed filtering from host polling changes. cpu-compact-* also filter IDs, enabling a matched native CPU baseline. CPU uses a serial native scan; CUDA uses a parallel atomic kernel",
            "buffer_reuse": "Replacement versus reuse includes allocation and preserving prior summary fault records by device-to-device copy; not allocation cost alone",
        },
        "rows": [],
    }
    runtime = (
        initialize_runtime(
            cuda_contexts=any(v.startswith("cuda-") for v in args.variants)
        )
        if any(v != "reference" for v in args.variants)
        else {"seconds": 0, "cuda_available": False, "devices": []}
    )
    report["runtime_initialization"] = runtime
    rng = random.Random(args.seed)
    failures = False
    preflight_cache = {}
    started = perf_counter()
    for fixture in args.fixtures:
        for batch in args.batches:
            for size in args.sizes:
                for shape in args.distributions:
                    construction_start = perf_counter()
                    try:
                        case = build_case(
                            fixture,
                            batch=batch,
                            size=size,
                            events=args.events,
                            distribution=shape,
                        )
                        if args.max_events is not None:
                            case = replace(case, max_events=args.max_events).validate()
                        if args.ordering == "grouped":
                            case = replace(
                                case,
                                scenarios=tuple(sorted(case.scenarios, key=digest)),
                            )
                    except ValueError as error:
                        report["rows"].append(
                            {
                                "fixture": fixture,
                                "batch": batch,
                                "size": size,
                                "distribution": shape,
                                "status": "unsupported_dimensions",
                                "error": str(error),
                            }
                        )
                        write_json(args.output, report)
                        continue
                    construction_seconds = perf_counter() - construction_start
                    case_hash = digest(case)
                    case_path = (
                        args.output.resolve().parent
                        / "fixtures"
                        / f"{fixture}-{case_hash[:16]}.json"
                    )
                    case.save(case_path)
                    try:
                        expected, work = reference_work(case)
                    except Exception as error:
                        report["rows"].append(
                            {
                                "fixture": fixture,
                                "batch": batch,
                                "size": size,
                                "distribution": shape,
                                "status": "reference_failed",
                                "reference_error": classify_error(error),
                            }
                        )
                        failures = True
                        write_json(args.output, report)
                        continue
                    for quantum in args.quanta:
                        config = dict(
                            quantum=quantum,
                            cpu_threads=cpu_threads,
                            cpu_chunk=args.cpu_chunk,
                            block_size=args.block_size,
                            poll_interval=args.poll_interval,
                            checkpoint_interval=args.checkpoint_interval,
                            cpu_affinity=topology["allowed_cpus"],
                            memory_limit=args.memory_mib * 1024**2,
                            mode=args.mode,
                        )
                        row = {
                            "fixture": fixture,
                            "description": FIXTURES[fixture],
                            "batch": batch,
                            "size": size,
                            "distribution": shape,
                            "ordering": args.ordering,
                            "config": config,
                            "case_sha256": case_hash,
                            "case_path": str(case_path),
                            "case_construction_seconds": construction_seconds,
                            "reference_work": work,
                            "reference_stop_quantum_histogram": {
                                status: dict(
                                    sorted(
                                        Counter(
                                            (result.events + quantum - 1) // quantum
                                            for result in expected
                                            if result.status == status
                                        ).items()
                                    )
                                )
                                for status in sorted(
                                    {result.status for result in expected}
                                )
                            },
                            "variants": {},
                            "orders": [],
                        }
                        report["rows"].append(row)
                        order = list(args.variants)
                        rng.shuffle(order)
                        row["warmup_order"] = order
                        for name in order:
                            variant = row["variants"][name] = {
                                "status": "ok",
                                "samples": [],
                            }
                            if (
                                name.startswith("cuda-")
                                and not runtime["cuda_available"]
                            ):
                                variant.update(
                                    status="unavailable_device",
                                    error="Warp reports no usable CUDA device",
                                )
                                continue
                            print(
                                f"Warmup {fixture}/{shape}: batch={batch}, size={size}, quantum={quantum}, {name}",
                                flush=True,
                            )
                            try:
                                if args.preflight_timeout and name != "reference":
                                    options = variant_options(name, config)
                                    key = digest(
                                        (
                                            case.program,
                                            case.trace_cells,
                                            options["device"],
                                            options["mode"],
                                            options["cpu_threads"] > 1,
                                            options["strategy"] != "scalar",
                                        )
                                    )
                                    if key in preflight_cache:
                                        variant["preflight_cache"] = {
                                            "compilation_key": key,
                                            "qualified_case_sha256": preflight_cache[
                                                key
                                            ],
                                            "scope": "same generated code already qualified; current case checked in warmup and every timed sample",
                                        }
                                    else:
                                        preflight = cold_sample(
                                            case_path,
                                            name,
                                            config,
                                            work["result_sha256"],
                                            args.preflight_timeout,
                                            isolated=False,
                                        )
                                        variant["preflight"] = preflight
                                        if preflight["status"] != "ok":
                                            variant.update(
                                                status="unavailable_preflight_timeout"
                                                if preflight["status"] == "timeout"
                                                else preflight["status"],
                                                error=preflight.get(
                                                    "error",
                                                    "Preflight did not complete",
                                                ),
                                            )
                                            failures |= variant["status"] == "error"
                                            continue
                                        preflight_cache[key] = case_hash
                                sample, actual = measure(case, name, config)
                                if actual != expected:
                                    raise AssertionError(
                                        "Warmup differs from independent reference"
                                    )
                                variant["warmup"] = sample
                            except Exception as error:
                                variant.update(classify_error(error))
                                failures |= variant["status"] == "error"
                        for repeat in range(args.repeats):
                            order = [
                                name
                                for name in args.variants
                                if row["variants"][name]["status"] == "ok"
                            ]
                            rng.shuffle(order)
                            row["orders"].append(order)
                            for name in order:
                                variant = row["variants"][name]
                                try:
                                    sample, actual = measure(case, name, config)
                                    if actual != expected:
                                        raise AssertionError(
                                            f"Measured repetition {repeat} differs from independent reference"
                                        )
                                    variant["samples"].append(sample)
                                except Exception as error:
                                    variant.update(classify_error(error))
                                    failures |= variant["status"] == "error"
                        for name, variant in row["variants"].items():
                            if variant["status"] != "ok":
                                continue
                            variant["summary"] = summarize(variant["samples"], work)
                            if not args.no_profile:
                                try:
                                    sample, actual = measure(
                                        case, name, config, profile=True
                                    )
                                    if actual != expected:
                                        raise AssertionError(
                                            "Instrumented sample differs from independent reference"
                                        )
                                    variant["instrumented_sample"] = sample
                                except Exception as error:
                                    variant["instrumented_sample"] = classify_error(
                                        error
                                    )
                                    failures |= (
                                        variant["instrumented_sample"]["status"]
                                        == "error"
                                    )
                            if args.cold:
                                variant["cold"] = cold_sample(
                                    case_path,
                                    name,
                                    config,
                                    work["result_sha256"],
                                    args.cold_timeout,
                                )
                                failures |= variant["cold"]["status"] in (
                                    "error",
                                    "timeout",
                                )
                        for baseline_name, field in (
                            ("cpu-serial", "median_speedup_vs_serial_cpu"),
                            (
                                "cpu-resident-serial",
                                "median_speedup_vs_resident_serial_cpu",
                            ),
                        ):
                            baseline = row["variants"].get(baseline_name, {})
                            if baseline.get("status") == "ok":
                                for variant in row["variants"].values():
                                    if variant["status"] == "ok":
                                        variant[field] = {
                                            key: baseline["summary"][key]["median"]
                                            / variant["summary"][key]["median"]
                                            for key in (
                                                "execution_and_materialization_seconds",
                                                "campaign_result_seconds",
                                                "campaign_with_checkpoint_restore_seconds",
                                            )
                                        }
                        write_json(args.output, report)
    report["benchmark_process_seconds"] = perf_counter() - started
    report["process_peak_rss_bytes"] = rss_bytes()
    report["source_unchanged_during_benchmark"] = hashes == source_hashes()
    report["status"] = (
        "error" if failures or not report["source_unchanged_during_benchmark"] else "ok"
    )
    write_json(args.output, report)
    print(f"Saved {args.output}; status={report['status']}", flush=True)
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
