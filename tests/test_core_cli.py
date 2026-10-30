"""Public common-core artifacts and CLI use the same closed execution path."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from warpsim.core import (
    BOOL,
    TIME,
    Bits,
    Case,
    Cell,
    Clock,
    Drive,
    Input,
    Program,
    Region,
    Require,
    Scenario,
    const,
    inp,
    now,
    op,
)
from warpsim.core.cli import example
from warpsim.core.serialization import load


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "warpsim.core", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "name", ["queued-fir", "fifo-pipeline"]
)
def test_all_models_export_closed_executable_cases(name, tmp_path):
    path = tmp_path / (name + ".json")
    case = example(name)
    case.save(path)
    assert Case.load(path) == case
    # Every exported model is executable through the same reference interface.
    result = Case.load(path).run(device="reference")
    assert result and all(r.status in ("horizon", "complete", "fault") for r in result)


def test_run_verify_compile_and_checkpoint_commands(tmp_path):
    case = tmp_path / "case.json"
    report = tmp_path / "report.json"
    checkpoint = tmp_path / "checkpoint.json"
    assert cli("example", "fifo-pipeline", case).returncode == 0
    checked = cli("verify", case, "--device", "cpu", "--output", report)
    assert checked.returncode == 0, checked.stderr
    assert load(report, "warpsim.report/v2")["matches"]
    run = cli("run", case, "--output", report, "--checkpoint", checkpoint)
    assert run.returncode == 0, run.stderr
    resumed = cli(
        "resume",
        checkpoint,
        "--until",
        "12",
        "--device",
        "reference",
        "--output",
        report,
    )
    assert resumed.returncode == 0, resumed.stderr
    source = tmp_path / "model.cpp"
    built = cli("compile", case, "--output", source)
    assert built.returncode == 0, built.stderr
    mapping = load(str(source) + ".map.json", "warpsim.source-map/v2")
    lines = source.read_text().splitlines()
    assert all(
        1 <= line <= len(lines)
        for positions in mapping["paths"].values()
        for line in positions
    )
    assert all(path in mapping["locations"] for path in mapping["paths"])
    assert 'extern "C" __global__ void advance' in source.read_text()


def test_failed_batch_replay_preserves_exact_time_lattice(tmp_path):
    # The failing lane has no fractional drives. Another lane refines the batch
    # lattice; TIME diagnostic operands must retain that same scale in replay.
    p = Program(
        "replay",
        (Cell("x", Bits(8)),),
        (Clock("tick"),),
        (Region("check", (Require(op("lt", now(), const(2)), "late", (now(),)),)),),
        inputs=(Input("value", Bits(8)),),
    )
    case = Case(p, "2", (Scenario(), Scenario((Drive("1/3", {"value": 7}),))))
    path, report = tmp_path / "case.json", tmp_path / "result.json"
    case.save(path)
    actual = cli("run", path, "--device", "cpu", "--output", report)
    assert actual.returncode == 3, actual.stderr
    results = load(report, "warpsim.report/v2")["results"]
    replay = Case.load(str(report) + ".replay.json")
    assert replay.time_denominator == 3
    assert replay.run(device="reference")[0] == results[0]
    checked = cli("verify", str(report) + ".replay.json", "--output", report)
    assert checked.returncode == 0, checked.stderr


def test_case_rejects_executable_constructors(tmp_path):
    path = tmp_path / "case.json"
    example("queued-fir").save(path)
    document = json.loads(path.read_text())
    # A checksum is integrity metadata, not permission to deserialize code.
    document["data"]["$type"] = "subprocess.Popen"
    import hashlib

    from warpsim.core.serialization import canonical

    document["sha256"] = hashlib.sha256(
        canonical(document["data"]).encode()
    ).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="constructor"):
        Case.load(path)
