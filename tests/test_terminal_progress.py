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
