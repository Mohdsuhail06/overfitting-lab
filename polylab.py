"""Polynomial regression from scratch, and the tools to catch overfitting (lesson 1.4).

  - poly_features      -> chapter 29: add x^2, x^3, ... as new columns
  - fit_poly           -> the same least squares as before, on those columns
  - cv_rmse_by_degree  -> chapter 34: choose the degree with cross-validation
  - learning_curve     -> chapter 33: how the error changes with more data
"""
import numpy as np
import pandas as pd

# Ages 0-22 are scaled to -1 .. 1 before raising to a power (chapter 29: x^10 gets huge).
CENTER, SCALE = 11.0, 11.0


def clean_cars(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning as the book: drop placeholder prices, keep ages 0-22 and prices <= 60,000."""
    fake = (df["price_JOD"] <= 1500) & (df["car_age"] <= 5)        # lesson 1.3 rule
    df = df[~fake].reset_index(drop=True)
    keep = (df["car_age"] >= 0) & (df["car_age"] <= 22) & (df["price_JOD"] <= 60000)
    return df[keep].reset_index(drop=True)


def poly_features(x: np.ndarray, degree: int, center: float = CENTER, scale: float = SCALE) -> np.ndarray:
    """Columns [1, z, z^2, ..., z^degree] with z = (x - center) / scale.

    The column of 1s is for b (chapter 13), the rest are the new "features".
    """
    z = (x - center) / scale
    return np.column_stack([z ** k for k in range(degree + 1)])


def fit_poly(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Least squares on the polynomial columns. Returns one weight per column.

    np.linalg.lstsq gives the same answer as the normal equation (chapter 14),
    but keeps working when columns are almost the same (high degrees).
    """
    weights, *_ = np.linalg.lstsq(poly_features(x, degree), y, rcond=None)
    return weights


def predict_poly(x: np.ndarray, weights: np.ndarray, degree: int) -> np.ndarray:
    """y_hat = b + w1*z + w2*z^2 + ...  (multiply each column by its weight and add)."""
    return poly_features(x, degree) @ weights


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Typical size of the error, in JOD."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def kfold_indices(n_rows: int, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """k pairs of (train_rows, test_rows); every row is tested exactly once."""
    pieces = np.array_split(np.random.default_rng(seed).permutation(n_rows), k)
    return [(np.concatenate([pieces[j] for j in range(k) if j != i]), pieces[i]) for i in range(k)]


def cv_rmse_by_degree(
    x: np.ndarray, y: np.ndarray, degrees: list[int], k: int = 5, seed: int = 0
) -> dict[int, tuple[float, float, float]]:
    """For every degree: (CV test RMSE mean, its spread, train RMSE mean)."""
    folds = kfold_indices(len(x), k, seed)
    result = {}
    for d in degrees:
        test_err, train_err = [], []
        for train, test in folds:
            w = fit_poly(x[train], y[train], d)
            train_err.append(rmse(y[train], predict_poly(x[train], w, d)))
            test_err.append(rmse(y[test], predict_poly(x[test], w, d)))
        result[d] = (float(np.mean(test_err)), float(np.std(test_err)), float(np.mean(train_err)))
    return result


def learning_curve(
    x: np.ndarray, y: np.ndarray, degree: int, sizes: list[int], repeats: int = 30, seed: int = 0
) -> list[tuple[int, float, float]]:
    """For every training size n: (n, train RMSE, test RMSE), each the median of `repeats` tries.

    Every try: pick n random cars to learn from, and test on all the other cars.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        tr, te = [], []
        for _ in range(repeats):
            order = rng.permutation(len(x))
            a, b = order[:n], order[n:]
            w = fit_poly(x[a], y[a], degree)
            tr.append(rmse(y[a], predict_poly(x[a], w, degree)))
            te.append(rmse(y[b], predict_poly(x[b], w, degree)))
        rows.append((n, float(np.median(tr)), float(np.median(te))))
    return rows
