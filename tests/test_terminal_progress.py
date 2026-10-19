from llama_benchy.terminal_progress import TerminalRunProgress, _format_duration


def test_first_eta_includes_warmup_and_only_measured_runs_count():
    current = [100.0]
    progress = TerminalRunProgress(4, clock=lambda: current[0])
    assert progress.start() == "[--------------------]   0% | 0/4 | Elapsed 00:00 | ETA --:--"
    current[0] += 8  # warmup, no progress
    assert progress.completed_runs == 0
    current[0] += 4  # first measured run
    assert progress.complete_run() == "[#####---------------]  25% | 1/4 | Elapsed 00:12 | ETA 00:36"
    current[0] += 12
    assert progress.complete_run() == "[##########----------]  50% | 2/4 | Elapsed 00:24 | ETA 00:24"
    current[0] += 12
    assert progress.complete_run() == "[###############-----]  75% | 3/4 | Elapsed 00:36 | ETA 00:12"
    current[0] += 12
    assert progress.complete_run() == "[####################] 100% | 4/4 | Elapsed 00:48 | ETA 00:00"
    assert progress.complete_run().startswith("[####################] 100% | 4/4")


def test_duration_formatting_and_zero_runs():
    assert _format_duration(3661) == "1:01:01"
    progress = TerminalRunProgress(0, clock=lambda: 0.0)
    assert progress.start() == "[--------------------]   0% | 0/0 | Elapsed 00:00 | ETA --:--"
    assert progress.complete_run() == progress.start()


def test_failed_runs_do_not_shorten_eta():
    current = [100.0]
    progress = TerminalRunProgress(4, clock=lambda: current[0])
    assert "ETA --:--" in progress.start()

    current[0] += 10
    assert "ETA 00:30" in progress.complete_run(success=True)

    current[0] += 0.1  # HTTP 400 rejected immediately
    second = progress.complete_run(success=False)
    assert "2/4" in second
    assert "ETA 00:20" in second
    assert "OK 1 FAIL 1" in second

    current[0] += 8
    third = progress.complete_run(success=True)
    assert "ETA 00:09" in third  # mean of 10s and 8s, not all attempted runs
    assert "OK 2 FAIL 1" in third

    current[0] += 0.1
    final = progress.complete_run(success=False)
    assert "100%" in final
    assert "ETA 00:00" in final
    assert "OK 2 FAIL 2" in final


def test_only_failed_runs_keep_eta_unknown_until_a_success():
    current = [0.0]
    progress = TerminalRunProgress(3, clock=lambda: current[0])
    progress.start()
    current[0] += 1
    assert "ETA --:--" in progress.complete_run(success=False)
    current[0] += 2
    assert "ETA --:--" in progress.complete_run(success=False)
    current[0] += 5
    assert "ETA 00:00" in progress.complete_run(success=False)
