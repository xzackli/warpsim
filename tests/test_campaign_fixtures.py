"""Independent small-workload expectations; no native timings in unit tests."""

from dataclasses import replace
from fractions import Fraction

import pytest

from warpsim.campaign_fixtures import FIXTURES, build_case
from warpsim.core import Case, Session


@pytest.mark.parametrize("name", FIXTURES)
def test_closed_case_and_portable_continuation(name, tmp_path):
    case = build_case(name, batch=4, size=7, events=5)
    path = tmp_path / "case.json"
    case.save(path)
    assert Case.load(path) == case
    expected = case.run(device="reference", trace_mode="summary")
    session = Session(
        case.program,
        case.scenarios,
        device="reference",
        trace_mode="summary",
        trace_cells=case.trace_cells,
    )
    session.step(until=case.until, max_events=2)
    path = tmp_path / "checkpoint.json"
    session.checkpoint(path)
    restored = Session.restore(path, device="reference")
    actual = restored.run(until=case.until, max_events=case.max_events - 2)
    assert actual == expected


def test_map_and_chain_match_independent_integer_recurrence():
    mask = (1 << 32) - 1
    width, ticks = 5, 4
    seed = 18  # identical-copy fixtures use the long-lived variant, number 15
    case = build_case(
        "map", batch=1, size=width, events=ticks, distribution="identical"
    )
    result = case.run(device="reference", trace_mode="summary")[0]
    values = list(range(width))
    for _ in range(ticks):
        values = [(x * 1664525 + seed) & mask for x in values]
    assert result.state == {"steps": (ticks,), "array": tuple(values), "bias": (seed,)}
    case = build_case(
        "chain", batch=1, size=width, events=ticks, distribution="identical"
    )
    result = case.run(device="reference", trace_mode="summary")[0]
    value = 1
    for _ in range(width * ticks):
        value = (value * 1664525 + seed) & mask
    assert result.state == {"steps": (ticks,), "value": (value,)}


def test_stencil_reads_one_region_snapshot():
    size, ticks = 7, 3
    case = build_case("stencil", batch=1, size=size, events=ticks)
    result = case.run(device="reference", trace_mode="summary")[0]
    values = list(range(size))
    for _ in range(ticks):
        values = [
            (values[(i - 1) % size] + 2 * values[i] + values[(i + 1) % size] + 3)
            % (1 << 32)
            for i in range(size)
        ]
    assert result.state["array"] == tuple(values)


def test_sparse_indirection_changes_only_visited_elements():
    size, ticks = 23, 9
    case = build_case("sparse", batch=1, size=size, events=ticks)
    result = case.run(device="reference", trace_mode="summary")[0]
    values, pointer = [0] * size, 0
    for _ in range(ticks):
        values[pointer] += 3
        pointer = (pointer + 17) % size
    assert result.state == {
        "steps": (ticks,),
        "array": tuple(values),
        "pointer": (pointer,),
    }


def test_timers_cancel_alternate_occurrences_and_preserve_final_pending_tag():
    ticks = 6
    case = build_case("timers", batch=1, size=1, events=ticks)
    result = case.run(device="reference")[0]
    sources = [r for r in result.records if r.observation == "sources"]
    assert sum(r.values[0] for r in sources) == ticks
    assert sum(r.values[1] for r in sources) == ticks - 1
    assert sum(r.values[2] for r in sources) == (ticks - 1) // 2
    assert result.events == ticks + ticks - 1 + (ticks - 1) // 2
    assert result.state["value"] == (3 * (ticks - 1) + (ticks - 1) // 2,)
    assert result.time == Fraction(ticks - 1, 3)
    assert result.calendar["timers"]["pulse"] == ((10, 1), ticks)
    assert result.calendar["timers"]["delayed"] == ((11, 0), ticks + (ticks + 1) // 2)


def test_skew_has_expected_faults_short_completions_and_one_survivor():
    case = build_case("skew", batch=16, size=1, events=40)
    results = case.run(device="reference", trace_mode="summary")
    assert [i for i, r in enumerate(results) if r.status == "fault"] == [2, 6, 10, 14]
    assert results[-1].events == 40
    for i, result in enumerate(results[:-1]):
        target = 1 + i % 4
        assert result.events == target
        assert result.state["steps"] == (target - int(result.status == "fault"),)
        if result.fault:
            assert result.fault.region == "update"
            assert result.fault.message == "expected skew fixture failure"
            assert result.fault.values[0] == target - 1


@pytest.mark.parametrize("name", FIXTURES)
def test_identical_and_diverse_batches_are_explicit(name):
    identical = build_case(name, batch=17, size=3, events=4, distribution="identical")
    diverse = build_case(name, batch=17, size=3, events=4, distribution="diverse")
    assert all(s == identical.scenarios[0] for s in identical.scenarios)
    assert diverse.scenarios[0] == diverse.scenarios[16]
    assert len({repr(s) for s in diverse.scenarios}) == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "unknown"},
        {"name": "control", "batch": 0},
        {"name": "control", "events": True},
        {"name": "sparse", "size": 65535},
        {"name": "map", "size": 1025},
        {"name": "chain", "size": 1024},
        {"name": "control", "distribution": "random"},
    ],
)
def test_unsupported_dimensions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_case(**kwargs)


def test_small_semantic_budget_is_a_real_resumable_outcome():
    case = replace(build_case("map", batch=2, size=4, events=12), max_events=3)
    results = case.run(device="reference", trace_mode="summary")
    assert all(r.status == "budget" and r.events == 3 for r in results)
