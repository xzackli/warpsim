"""Run and verify the reusable-core example: python examples/queued_fir.py --help."""

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path

from warpsim.core import Drive, run_reference, run_native
from warpsim.models.queued_fir import queued_fir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("reference", "cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = queued_fir()
    drives = (Drive("0", {"valid": 1, "value": 3, "id": 1}),
              Drive("1", {"value": 5, "id": 2, "source_time": 1}),
              Drive("2", {"valid": 0, "ready": 0}),
              Drive("4", {"ready": 1}))
    options = dict(until="6", drives=drives, max_events=16)
    expected = run_reference(model, **options)
    result = expected if args.device == "reference" else run_native(model, device=args.device, **options)
    report = {"schema": "warpsim.core.example/v1", "device": args.device,
              "matches_reference": result == expected, "program": asdict(model),
              "inputs": [asdict(d) for d in drives], "until": "6", "max_events": 16,
              "result": asdict(result)}

    def exact(value):
        if isinstance(value, Fraction):
            return str(value)
        raise TypeError(type(value).__name__)

    encoded = json.dumps(report, default=exact, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"{args.device}: {result.status}; {result.events} event groups; matches reference: {result == expected}")
        print(args.output)
    else:
        print(encoded, end="")
    return 0 if result == expected and result.status == "horizon" else 1


if __name__ == "__main__":
    raise SystemExit(main())
