"""Tests for polylab.py. Run with: python -m pytest -v"""
import numpy as np
import pandas as pd
import pytest

from polylab import (
    clean_cars, cv_rmse_by_degree, fit_poly, kfold_indices, learning_curve,
    poly_features, predict_poly, rmse,
)


def test_poly_features_book_example():
    # Chapter 29: age 10 -> x = 10, x^2 = 100, x^3 = 1,000 (no scaling here).
    table = poly_features(np.array([10.0]), degree=3, center=0.0, scale=1.0)
    assert table.tolist() == [[1.0, 10.0, 100.0, 1000.0]]   # first column of 1s is for b


def test_poly_features_scaled_to_minus_one_one():
    table = poly_features(np.array([0.0, 11.0, 22.0]), degree=2, center=11.0, scale=11.0)
    assert table[:, 1].tolist() == [-1.0, 0.0, 1.0]
    assert table[:, 2].tolist() == [1.0, 0.0, 1.0]


def test_fit_recovers_a_perfect_curve():
    x = np.linspace(0, 20, 30)
    y = 3 * x ** 2 - 40 * x + 500
    w = fit_poly(x, y, degree=2)
    assert predict_poly(np.array([7.0]), w, degree=2)[0] == pytest.approx(3 * 49 - 280 + 500)


def test_degree_n_minus_1_goes_through_every_point():
    # 12 points with different x: degree 11 can hit all of them exactly (chapter 30).
    rng = np.random.default_rng(0)
    x = np.arange(0, 23, 2, dtype=float)
    y = rng.uniform(2000, 20000, size=12)
    w = fit_poly(x, y, degree=11)
    assert rmse(y, predict_poly(x, w, degree=11)) == pytest.approx(0.0, abs=1e-3)


def test_clean_cars_rule():
    cars = pd.DataFrame({"car_age": [1, 30, 3, 25, -1],
                         "price_JOD": [1000, 1000, 15000, 70000, 9000]})
    clean = clean_cars(cars)
    # removed: new & cheap (1000 at age 1), age outside 0-22 (30, 25, -1)
    assert clean["price_JOD"].tolist() == [15000]


def test_kfold_covers_every_row_once():
    folds = kfold_indices(17, k=4, seed=1)
    tested = sorted(np.concatenate([t for _, t in folds]).tolist())
    assert tested == list(range(17))


def test_cv_prefers_the_true_degree():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 22, 400)
    y = 30 * x ** 2 - 1200 * x + 18000 + rng.normal(0, 500, 400)   # a real curve
    scores = cv_rmse_by_degree(x, y, degrees=[1, 2], k=5, seed=0)
    assert scores[2][0] < scores[1][0]        # degree 2 beats a straight line


def test_learning_curve_shape():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 22, 500)
    y = -700 * x + 18000 + rng.normal(0, 2000, 500)
    rows = learning_curve(x, y, degree=1, sizes=[20, 200], repeats=5, seed=0)
    assert [r[0] for r in rows] == [20, 200]
    assert all(len(r) == 3 for r in rows)
