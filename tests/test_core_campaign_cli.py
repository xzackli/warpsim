"""Public CLI qualification of resident scheduling and strategy reports."""

import subprocess
import sys

import pytest
import warp as wp

from warpsim.core import (
    BOOL,
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
    Session,
    Store,
    const,
    inp,
    op,
    read,
)
from warpsim.core.serialization import load


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "warpsim.core", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def campaign_case():
    byte, word = Bits(8), Bits(32)
    program = Program(
        "cli_campaign",
        (Cell("source", byte, 33), Cell("output", word, 33), Cell("count", word)),
        (Clock("tick", "1/4"),),
        (
            Region(
                "prefix",
                (
                    Store("count", op("add", read("count", word), const(1, word))),
                    Require(op("not", inp("fail", BOOL)), "requested lane failure"),
                ),
            ),
            Region(
                "map",
                tuple(
                    Store(
                        "output",
                        op(
                            "add",
                            op("cast", read("source", byte, const(i)), type=word),
                            const(1, word),
                        ),
                        const(i),
                    )
                    for i in range(33)
                ),
            ),
        ),
        inputs=(Input("halt", BOOL), Input("fail", BOOL)),
        stop=inp("halt", BOOL),
    )
    drives = (
        (Drive("0", {"halt": 1}),),
        (Drive("1/2", {"fail": 1}),),
        (Drive("3/4", {"halt": 1}),),
        (),
    )
    scenarios = tuple(
        Scenario(schedule, {"source": [(i * 19 + lane * 7) % 256 for i in range(33)]})
        for lane, schedule in enumerate(drives)
    )
    return Case(program, "1", scenarios, max_events=20, trace_cells=("count", "output"))


def test_resident_multicore_report_checkpoint_and_failure_replay(tmp_path):
    case = campaign_case()
    source = tmp_path / "case.json"
    output = tmp_path / "report.json"
    checkpoint = tmp_path / "checkpoint.json"
    case.save(source)
    completed = cli(
        "run",
        source,
        "--output",
        output,
        "--checkpoint",
        checkpoint,
        "--resident",
        "--trace-mode",
        "summary",
        "--compact",
        "--quantum",
        1,
        "--poll-interval",
        2,
        "--cpu-threads",
        2,
        "--cpu-schedule",
        "dynamic",
        "--cpu-chunk",
        2,
        "--no-buffer-reuse",
        "--profile",
        "--mode",
        wp.config.mode,
    )
    assert completed.returncode == 3, completed.stderr
    report = load(output, "warpsim.report/v2")
    assert report["results"] == case.run(device="reference", trace_mode="summary")
    assert report["strategy"] == "scalar"
    assert report["execution_metrics"]["launches"] > 1
    assert (
        report["execution_metrics"]["trace_allocations"]
        == report["execution_metrics"]["launches"]
    )
    assert report["execution_metrics"]["host_readback_bytes"] > 0
    replay = Case.load(report["replay"])
    assert replay.scenarios == (case.scenarios[1],)
    assert (
        replay.run(device="reference", trace_mode="summary")[0] == report["results"][1]
    )
    expected = Session.restore(checkpoint, device="reference").run(
        until="2", max_events=10
    )
    resumed = cli(
        "resume",
        checkpoint,
        "--output",
        output,
        "--until",
        "2",
        "--max-events",
        10,
        "--resident",
        "--compact",
        "--quantum",
        2,
        "--cpu-threads",
        2,
        "--cpu-schedule",
        "static",
        "--profile",
        "--mode",
        wp.config.mode,
    )
    assert resumed.returncode == 3, resumed.stderr
    assert load(output, "warpsim.report/v2")["results"] == expected


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is required")
@pytest.mark.parametrize("strategy", ("scalar", "warp", "block"))
def test_cli_resident_cuda_strategy_verification_compares_complete_results(
    strategy, tmp_path
):
    case = campaign_case()
    source, output = tmp_path / "case.json", tmp_path / "report.json"
    case.save(source)
    result = cli(
        "verify",
        source,
        "--output",
        output,
        "--device",
        "cuda:0",
        "--resident",
        "--trace-mode",
        "summary",
        "--strategy",
        strategy,
        "--compact",
        "--quantum",
        1,
        "--block-size",
        64,
        "--profile",
        "--mode",
        wp.config.mode,
    )
    assert result.returncode == 0, result.stderr
    report = load(output, "warpsim.report/v2")
    assert report["matches"] is True
    assert report["strategy"] == strategy
    assert report["results"] == case.run(device="reference", trace_mode="summary")


@pytest.mark.parametrize(
    "options,reason",
    (
        (("--resident",), "summary"),
        (("--compact",), "resident"),
        (("--cpu-threads", "0"), "Integer"),
        (("--resident", "--trace-mode", "summary", "--device", "reference"), "native"),
    ),
)
def test_cli_rejects_invalid_execution_contract_without_writing_report(
    options, reason, tmp_path
):
    source, output = tmp_path / "case.json", tmp_path / "report.json"
    campaign_case().save(source)
    result = cli("run", source, "--output", output, *options)
    assert result.returncode == 2
    assert reason.lower() in result.stderr.lower()
    assert not output.exists()
