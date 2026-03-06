import math

from app.service import _robust_padded_range


def test_robust_range_ignores_extreme_tail_with_four_models():
    outputs = [
        {"prediction_log": 2.0},
        {"prediction_log": 2.1},
        {"prediction_log": 2.2},
        {"prediction_log": 6.0},
    ]

    r = _robust_padded_range(outputs)

    assert math.isclose(r["core_min_log"], 2.1, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(r["core_max_log"], 2.2, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(r["min_log"], 1.95, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(r["max_log"], 2.35, rel_tol=0.0, abs_tol=1e-9)
    assert r["max_log"] < 6.0
    assert r["max_raw"] >= r["min_raw"] >= 0.0


def test_robust_range_falls_back_for_small_ensembles():
    outputs = [
        {"prediction_log": 1.0},
        {"prediction_log": 1.3},
    ]

    r = _robust_padded_range(outputs)

    assert math.isclose(r["core_min_log"], 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(r["core_max_log"], 1.3, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(r["min_log"], 0.75, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(r["max_log"], 1.55, rel_tol=0.0, abs_tol=1e-9)
    assert r["max_raw"] >= r["min_raw"] >= 0.0
