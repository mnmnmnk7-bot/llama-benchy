import pytest

from llama_benchy.client import RequestResult
from llama_benchy.results import BenchmarkResults


def test_decode_throughput_uses_observed_token_interval():
    result = RequestResult(
        start_ts=0.0,
        first_response_ts=1.0,
        first_token_ts=1.0,
        end_ts=1.35,
        prompt_tokens=100,
        total_tokens=4,
        token_timestamps=[1.0, 1.1, 1.2, 1.3],
    )

    results = BenchmarkResults()
    results.add("model", 100, 4, 0, 1, [[result]], latency=0.0, expected_pp_tokens=100)

    assert results.runs[0].tg_throughput is not None
    assert results.runs[0].tg_throughput.mean == pytest.approx(10.0)


def test_burst_output_does_not_report_decode_throughput():
    result = RequestResult(
        start_ts=0.0,
        first_response_ts=3.0,
        first_token_ts=3.0,
        end_ts=3.001,
        prompt_tokens=2048,
        total_tokens=130,
        token_timestamps=[3.0] * 130,
    )

    results = BenchmarkResults()
    results.add("model", 2048, 1024, 0, 1, [[result]], latency=0.0, expected_pp_tokens=2048)

    run = results.runs[0]
    assert run.tg_throughput is None
    assert run.peak_throughput is not None
    assert run.peak_throughput.mean == pytest.approx(130.0)

    rows = results._generate_rows()
    tg_row = next(row for row in rows if row["test_name"] == "tg1024")
    assert tg_row["t_s"] is None
    assert tg_row["peak_ts"] is not None


def test_block_streaming_excludes_first_observed_block_from_decode_throughput():
    first_block = [1.0] * 256
    second_block = [1.25 + (0.75 * (i + 1) / 256) for i in range(256)]

    result = RequestResult(
        start_ts=0.0,
        first_response_ts=0.9,
        first_token_ts=1.0,
        end_ts=2.01,
        prompt_tokens=2048,
        total_tokens=512,
        token_timestamps=first_block + second_block,
    )

    results = BenchmarkResults()
    results.add("model", 2048, 512, 0, 1, [[result]], latency=0.0, expected_pp_tokens=2048)

    run = results.runs[0]
    assert run.tg_throughput is not None
    assert run.tg_throughput.mean == pytest.approx(256.0)


def _successful_request():
    return RequestResult(
        start_ts=0.0,
        first_response_ts=1.0,
        first_token_ts=1.0,
        end_ts=1.3,
        prompt_tokens=100,
        total_tokens=4,
        token_timestamps=[1.0, 1.1, 1.2, 1.3],
    )


def test_partial_concurrency_batch_is_excluded_from_performance_metrics():
    baseline = BenchmarkResults()
    baseline.add(
        "model", 100, 4, 0, 2,
        [[_successful_request(), _successful_request()]],
        latency=0.0, expected_pp_tokens=100,
    )
    results = BenchmarkResults()
    results.add(
        "model", 100, 4, 0, 2,
        [
            [_successful_request(), _successful_request()],
            [_successful_request(), RequestResult(error="HTTP 400: context too long")],
        ],
        latency=0.0, expected_pp_tokens=100,
    )
    run = results.runs[0]
    assert (run.total_runs, run.successful_runs, run.failed_runs) == (2, 1, 1)
    assert run.status == "partial"
    assert "HTTP 400" in run.error
    assert run.tg_throughput.mean == pytest.approx(baseline.runs[0].tg_throughput.mean)
    assert run.pp_throughput.mean == pytest.approx(baseline.runs[0].pp_throughput.mean)


def test_total_failure_exports_status_and_empty_metrics(tmp_path):
    import csv

    results = BenchmarkResults()
    results.add(
        "model", 2048, 128, 65535, 2,
        [[RequestResult(error="HTTP 400: max_tokens=0"),
          RequestResult(error="HTTP 400: max_tokens=0")]],
        latency=0.0,
        expected_pp_tokens=65535,
        is_context_phase=True,
    )
    run = results.runs[0]
    assert (run.total_runs, run.successful_runs, run.failed_runs) == (1, 0, 1)
    assert run.status == "failed"
    assert run.pp_throughput is None
    assert run.tg_throughput is None
    assert run.peak_throughput is None

    target = tmp_path / "result.csv"
    results.save_report(str(target), "csv", concurrency=2)
    with target.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["test_name"].split(" @ ")[0] for row in rows} == {"ctx_pp", "ctx_tg"}
    assert all(row["status"] == "failed" for row in rows)
    assert all(row["successful_runs"] == "0" for row in rows)
    assert all(row["failed_runs"] == "1" for row in rows)
    assert all(not row["t_s_mean"] for row in rows)
    assert all(not row["peak_ts_mean"] for row in rows)
    assert all("HTTP 400" in row["error"] for row in rows)


def test_no_output_response_is_treated_as_invalid_batch():
    results = BenchmarkResults()
    results.add(
        "model", 100, 4, 0, 1,
        [[RequestResult(start_ts=1, end_ts=2, error=None, total_tokens=0)]],
        latency=0.0, expected_pp_tokens=100,
    )
    run = results.runs[0]
    assert run.status == "failed"
    assert "without generated tokens" in run.error
    assert run.tg_throughput is None


def test_single_generated_token_does_not_become_peak_decode_throughput():
    results = BenchmarkResults()
    results.add(
        "model", 100, 1, 0, 1,
        [[RequestResult(
            start_ts=0.1, first_response_ts=0.9, first_token_ts=1.0,
            end_ts=1.1, prompt_tokens=100, total_tokens=1,
            token_timestamps=[1.0],
        )]],
        latency=0.0, expected_pp_tokens=100,
    )
    run = results.runs[0]
    assert run.status == "ok"
    assert run.tg_throughput is None
    assert run.peak_throughput is None
    assert not any(row["test_name"].startswith("tg1") for row in results._generate_rows())
