"""Create a composed traffic case for the unmodified common-core CLI."""

import argparse

from warpsim.core import Case, Drive, Scenario
from warpsim.models.traffic import traffic_signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="closed semantic case JSON")
    args = parser.parse_args()
    case = Case(
        traffic_signals().program,
        "6",
        (
            Scenario(),
            Scenario((Drive("1", {"emergency": 1}), Drive("3/2", {"emergency": 0}))),
        ),
    )
    case.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
