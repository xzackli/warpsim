"""A different application crosses the existing public architecture boundaries."""

import subprocess
import sys
from dataclasses import replace
from fractions import Fraction

import pytest
import warp as wp

from warpsim.core import Case, Drive, Scenario, Session, run_native, run_reference
from warpsim.core.codegen import Emitter
from warpsim.core.lower import lower
from warpsim.modeling import verify_schedule
from warpsim.models.traffic import traffic_signals


DEVICES = ["reference", "cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


def observations(result, name):
    return [r for r in result.records if r.observation == name]


@pytest.mark.parametrize("device", DEVICES)
def test_rational_phase_cycle_publication_and_independent_interlock(device):
    composition = traffic_signals()
    expected = run_reference(composition.program, until="6")
    actual = (
        expected
        if device == "reference"
        else run_native(
            composition.program, until="6", device=device, mode=wp.config.mode
        )
    )
    assert actual == expected
    assert actual.status == "horizon" and actual.fault is None
    assert verify_schedule(composition.program, composition.certificate)
    # Hand-calculated phase durations, independent of the declaration builder.
    times = list(map(Fraction, ("1/3", "7/3", "17/6", "19/6", "31/6", "17/3", "6")))
    phases = (0, 1, 2, 3, 4, 5, 0)
    assert [
        (r.time, r.microstep, r.values)
        for r in observations(actual, "controller__phase")
    ] == [(time, 0, (phase,)) for time, phase in zip(times, phases)]
    colors = ((0, 0), (2, 0), (1, 0), (0, 0), (0, 2), (0, 1), (0, 0), (2, 0))
    for direction, column in (("ns", 0), ("ew", 1)):
        assert [
            (r.time, r.microstep, r.values)
            for r in observations(actual, direction + "__color")
        ] == [
            (time, 1, (pair[column],))
            for time, pair in zip([Fraction(0), *times], colors)
        ]
    checks = observations(actual, "monitor__checked")
    assert [(r.time, r.microstep, r.values) for r in checks] == [
        (time, 2, tuple(int(color == 2) for color in pair))
        for time, pair in zip([Fraction(0), *times], colors)
    ]
    assert actual.state["monitor__checks"] == (8,)
    assert all(
        r.state["ns__color"] != (2,) or r.state["ew__color"] != (2,)
        for r in actual.records
    )
    emitter = Emitter(lower(composition.program))
    lines = emitter.generate().splitlines()
    for record in checks:
        origin = composition.origin(record.location)
        assert (origin.instance, origin.component, origin.action) == (
            "monitor",
            "interlock_monitor",
            "check",
        )
        assert all(
            1 <= line <= len(lines) for line in emitter.source_map[record.location]
        )


@pytest.mark.parametrize("device", DEVICES)
def test_sampled_emergency_overrides_coincident_expiry_and_restarts_clearance(device):
    scenarios = (
        Scenario(),
        Scenario((Drive("1/3", {"emergency": 1}), Drive("4/3", {"emergency": 0}))),
        Scenario((Drive("1", {"emergency": 1}), Drive("3/2", {"emergency": 0}))),
    )
    case = Case(traffic_signals().program, "2", scenarios)
    actual = case.run(device=device, mode=wp.config.mode)
    assert actual == case.run(device="reference")
    expected_phases = (
        ((Fraction(1, 3), (0,)),),
        ((Fraction(5, 3), (0,)),),
        ((Fraction(1, 3), (0,)), (Fraction(2), (0,))),
    )
    expected_emergencies = (
        (),
        ((Fraction(1, 3), (1,)), (Fraction(4, 3), (0,))),
        ((Fraction(1), (1,)), (Fraction(5, 3), (0,))),
    )
    for result, phases, emergencies in zip(
        actual, expected_phases, expected_emergencies
    ):
        assert (
            tuple((r.time, r.values) for r in observations(result, "controller__phase"))
            == phases
        )
        assert (
            tuple(
                (r.time, r.values)
                for r in observations(result, "controller__emergency")
            )
            == emergencies
        )
        assert result.fault is None
        assert all(r.values != (1, 1) for r in observations(result, "monitor__checked"))


@pytest.mark.parametrize("origin", DEVICES)
@pytest.mark.parametrize("target", DEVICES)
def test_checkpoint_between_controller_commit_and_head_publication(
    origin, target, tmp_path
):
    scenarios = (
        Scenario(),
        Scenario((Drive("1/3", {"emergency": 1}), Drive("4/3", {"emergency": 0}))),
    )
    case = Case(traffic_signals().program, "6", scenarios)
    session = Session(case.program, scenarios, device=origin, mode=wp.config.mode)
    prefix = session.step(until="6", max_events=4)
    assert prefix[0].time == Fraction(1, 3) and prefix[0].microstep == 0
    assert prefix[0].state["controller__phase"] == (0,) and prefix[0].state[
        "ns__color"
    ] == (0,)
    assert prefix[0].calendar["timers"]["controller__publish"][0][1] == 1
    checkpoint = tmp_path / "signals.checkpoint.json"
    session.checkpoint(checkpoint)
    suffix = Session.restore(checkpoint, device=target, mode=wp.config.mode).run(
        until="6"
    )
    combined = tuple(
        replace(after, records=before.records + after.records)
        for before, after in zip(prefix, suffix)
    )
    assert combined == case.run(device="reference")


@pytest.mark.parametrize("device", DEVICES)
def test_changed_phase_durations_are_model_data(device):
    program = traffic_signals(green="1/2", amber="1/4", clearance="1/8").program
    actual = Case(program, "1").run(device=device, mode=wp.config.mode)[0]
    assert [(r.time, r.values) for r in observations(actual, "controller__phase")] == [
        (Fraction(1, 8), (0,)),
        (Fraction(5, 8), (1,)),
        (Fraction(7, 8), (2,)),
        (Fraction(1), (3,)),
    ]


def test_closed_case_cli_needs_no_declaration_imports(tmp_path):
    case = tmp_path / "signals.json"
    Case(traffic_signals().program, "6").save(case)
    script = r"""
import importlib.abc
import sys
from dataclasses import replace
from pathlib import Path

class NoApplications(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("warpsim.models", "warpsim.modeling", "warpsim.components")
        if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
            raise AssertionError("Generic path imported application code: " + fullname)

sys.meta_path.insert(0, NoApplications())
from warpsim.core import Case
from warpsim.core.cli import main
from warpsim.core.serialization import load

case, mode = Path(sys.argv[1]), sys.argv[2]
source, report, checkpoint = (case.with_suffix(s) for s in (".cpp", ".report.json", ".checkpoint.json"))
assert main(["compile", str(case), "--output", str(source)]) == 0
assert load(str(source) + ".map.json", "warpsim.source-map/v2")["paths"]
assert main(["verify", str(case), "--output", str(report), "--mode", mode]) == 0
assert load(report, "warpsim.report/v2")["matches"]
replace(Case.load(case), max_events=4).save(case)
assert main(["run", str(case), "--output", str(report), "--checkpoint", str(checkpoint), "--mode", mode]) == 4
assert main(["resume", str(checkpoint), "--until", "6", "--output", str(report), "--mode", mode]) == 0
result = load(report, "warpsim.report/v2")["results"][0]
assert result.state["monitor__checks"] == (8,)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(case), wp.config.mode],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
