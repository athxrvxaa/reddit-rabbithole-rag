from rabbithole.eval import _bootstrap_mean_interval, intervals_overlap, wilson_interval


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(5, 10)
    assert low < .5 < high
    assert wilson_interval(0, 10)[0] == 0


def test_bootstrap_is_reproducible():
    assert _bootstrap_mean_interval([0, .5, 1], seed=4) == _bootstrap_mean_interval([0, .5, 1], seed=4)


def test_detect_overlapping_mode_intervals():
    assert intervals_overlap({"a": {"r_ci95": [0, .6]}, "b": {"r_ci95": [.4, 1]}}, "r")
    assert not intervals_overlap({"a": {"r_ci95": [0, .2]}, "b": {"r_ci95": [.8, 1]}}, "r")
