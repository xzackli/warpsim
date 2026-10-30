"""File-oriented entry point for the reusable semantic compiler and executor."""

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

from .ir import Drive
from .serialization import Case, save
from .session import Scenario, Session


def parser():
    p = argparse.ArgumentParser(
        prog="vlsi-sim core",
        description="Compile and run closed transition programs on reference, CPU or CUDA",
    )
    sub = p.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("example", help="write a self-contained semantic case")
    ex.add_argument(
        "name", choices=("queued-fir", "fifo-pipeline")
    )
    ex.add_argument("output", type=Path)
    for command in ("run", "verify", "resume", "compile"):
        q = sub.add_parser(command)
        q.add_argument("input", type=Path)
        q.add_argument("--output", type=Path, required=True)
        if command == "compile":
            continue
        q.add_argument("--device", default="cpu")
        q.add_argument("--mode", choices=("release", "debug"), default="release")
        q.add_argument("--memory-mib", type=int, default=256)
        q.add_argument(
            "--trace-mode", choices=("full", "observations", "summary"), default="full"
        )
        q.add_argument("--checkpoint", type=Path)
        q.add_argument("--cpu-threads", type=int, default=1)
        q.add_argument(
            "--cpu-schedule", choices=("static", "dynamic"), default="static"
        )
        q.add_argument("--cpu-chunk", type=int, default=1)
        q.add_argument(
            "--strategy", choices=("scalar", "warp", "block"), default="scalar"
        )
        q.add_argument(
            "--block-size", type=int, choices=(1, 32, 64, 128, 256), default=128
        )
        q.add_argument(
            "--resident",
            action="store_true",
            help="summary campaign with explicit final inspection",
        )
        q.add_argument(
            "--quantum",
            type=int,
            default=1000,
            help="event groups per scheduling yield",
        )
        q.add_argument(
            "--compact",
            action="store_true",
            help="compact active IDs between resident advances",
        )
        q.add_argument("--poll-interval", type=int, default=1)
        q.add_argument("--no-buffer-reuse", action="store_true")
        q.add_argument(
            "--profile",
            action="store_true",
            help="record instrumented execution timings",
        )
        if command == "resume":
            q.add_argument("--until", required=True, help="exact rational horizon")
            q.add_argument("--max-events", type=int, default=256)
    return p


def example(name):
    if name == "queued-fir":
        from ..models.queued_fir import queued_fir

        return Case(
            queued_fir(),
            "8",
            (
                Scenario(
                    (
                        Drive("0", {"valid": 1, "value": 3, "id": 7}),
                        Drive("1", {"valid": 0}),
                    )
                ),
            ),
        )
    if name == "fifo-pipeline":
        from ..models.fifo_pipeline import fifo_pipeline

        return Case(
            fifo_pipeline((2, 3, 1)).program,
            "8",
            (
                Scenario(
                    (
                        Drive("0", {"valid": 1, "value": -7, "id": 10}),
                        Drive("1", {"valid": 0}),
                    )
                ),
            ),
        )
    raise ValueError(f"Unknown example: {name}")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "example":
            example(args.name).save(args.output)
            print(json.dumps({"case": str(args.output)}))
            return 0
        if args.command == "compile":
            from .codegen import Emitter
            from .lower import lower

            case = Case.load(args.input)
            lowered = lower(case.program)
            emitter = Emitter(lowered, case.trace_cells)
            source = emitter.generate()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(source)
            save(
                str(args.output) + ".map.json",
                "warpsim.source-map/v2",
                {
                    "model_hash": lowered.digest,
                    "paths": emitter.source_map,
                    "locations": lowered.locations,
                    "effects": {
                        name: {
                            k: tuple(sorted(v)) if isinstance(v, set) else v
                            for k, v in e.items()
                        }
                        for name, e in lowered.effects.items()
                    },
                },
            )
            print(
                json.dumps({"source": str(args.output), "model_hash": lowered.digest})
            )
            return 0
        options = dict(
            device=args.device,
            mode=args.mode,
            memory_limit=args.memory_mib * 1024**2,
            cpu_threads=args.cpu_threads,
            cpu_schedule=args.cpu_schedule,
            cpu_chunk=args.cpu_chunk,
            strategy=args.strategy,
            block_size=args.block_size,
            reuse_buffers=not args.no_buffer_reuse,
            profile=args.profile,
        )
        if args.compact and not args.resident:
            raise ValueError("--compact requires --resident")
        if args.command == "resume":
            session = Session.restore(args.input, **options)
            until, budget = args.until, args.max_events
        else:
            case = Case.load(args.input)
            # Verification compares complete semantic observations and faults.
            trace_mode = (
                "full"
                if args.command == "verify" and not args.resident
                else args.trace_mode
            )
            session = Session(
                case.program,
                case.scenarios,
                trace_cells=case.trace_cells,
                trace_mode=trace_mode,
                time_denominator=case.time_denominator,
                **options,
            )
            until, budget = case.until, case.max_events
        results = (
            session.campaign(
                until=until,
                max_events=budget,
                quantum=args.quantum,
                compact=args.compact,
                poll_interval=args.poll_interval,
            )
            if args.resident
            else session.run(until=until, max_events=budget)
        )
        report = {
            "results": results,
            "device": args.device,
            "mode": args.mode,
            "until": until,
            "model_hash": session.l.digest,
            "execution_metrics": {
                key: Fraction(str(value)) if isinstance(value, float) else value
                for key, value in session.metrics.items()
            },
            "strategy": session.strategy,
        }
        code = (
            3
            if any(r.status == "fault" for r in results)
            else 4
            if any(r.status == "budget" for r in results)
            else 0
        )
        if args.command == "verify":
            expected = case.run(
                device="reference",
                memory_limit=options["memory_limit"],
                trace_mode=trace_mode,
            )
            report["matches"] = results == expected
            code = 0 if report["matches"] else 5
        if args.checkpoint:
            session.checkpoint(args.checkpoint)
        failures = [i for i, r in enumerate(results) if r.fault]
        if failures:
            # Stable scenario order; replay starts from original explicit inputs.
            i = failures[0]
            replay = Case(
                session.program,
                until,
                (session.scenarios[i],),
                results[i].events,
                session.trace_cells,
                session.denominator,
            )
            replay.save(str(args.output) + ".replay.json")
            report["replay"] = str(args.output) + ".replay.json"
        save(args.output, "warpsim.report/v2", report)
        print(
            json.dumps(
                {
                    "statuses": [r.status for r in results],
                    "report": str(args.output),
                    **({"matches": report["matches"]} if "matches" in report else {}),
                }
            )
        )
        return code
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
