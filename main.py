"""Overfitting lab: watch a model memorize, then choose the degree honestly.

  1. 12 cars (one per even age) -> fit degrees 0..11, test on all other cars.
  2. All cars -> 5-fold cross-validation for degrees 1..10, pick the best degree.
  3. Learning curves: degree 2 vs degree 9 with 15 ... 3,000 training cars.

Run:  python main.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polylab import (
    clean_cars, cv_rmse_by_degree, fit_poly, learning_curve, predict_poly, rmse,
)

HERE = Path(__file__).parent
IMAGES = HERE / "images"
IMAGES.mkdir(exist_ok=True)


def main() -> None:
    df = clean_cars(pd.read_csv(HERE / "data" / "cars_age_price.csv"))
    x = df["car_age"].to_numpy(dtype=float)
    y = df["price_JOD"].to_numpy(dtype=float)
    print(f"Cars after cleaning (age 0-22): {len(df):,}\n")

    # 1. Twelve cars: one random car for every even age (same pick as the book, seed 7)
    rng = np.random.default_rng(7)
    small = np.array([rng.choice(df.index[df["car_age"] == a]) for a in range(0, 23, 2)])
    rest = np.setdiff1d(np.arange(len(df)), small)
    print("1) Learn from 12 cars, test on the other", f"{len(rest):,}")
    print(f"   {'degree':>6}{'train RMSE':>12}{'test RMSE':>12}")
    small_rows = []
    for d in range(12):
        w = fit_poly(x[small], y[small], d)
        tr = rmse(y[small], predict_poly(x[small], w, d))
        te = rmse(y[rest], predict_poly(x[rest], w, d))
        small_rows.append((d, tr, te, w))
        print(f"   {d:>6}{tr:>12,.0f}{te:>12,.0f}")
    best_small = min(small_rows, key=lambda r: r[2])[0]
    print(f"   -> best on the test cars: degree {best_small}; degree 11 memorized (train error 0)\n")

    # 2. All cars: choose the degree with 5-fold cross-validation
    print("2) All cars, 5-fold cross-validation")
    cv = cv_rmse_by_degree(x, y, degrees=list(range(1, 11)), k=5, seed=0)
    print(f"   {'degree':>6}{'train RMSE':>12}{'CV RMSE':>12}{'+/-':>7}")
    for d, (mean, std, tr) in cv.items():
        print(f"   {d:>6}{tr:>12,.0f}{mean:>12,.0f}{std:>7,.0f}")
    best = min(cv, key=lambda d: cv[d][0])
    # Simplest degree whose CV error is within 1% of the best (chapter 34: prefer the simpler model)
    simple = min(d for d in cv if cv[d][0] <= cv[best][0] * 1.01)
    print(f"   -> lowest CV error: degree {best}; simplest within 1% of it: degree {simple}\n")

    # 3. Learning curves
    sizes = [15, 30, 60, 120, 250, 500, 1000, 3000]
    lc = {d: learning_curve(x, y, d, sizes, repeats=30, seed=0) for d in (2, 9)}
    print("3) Learning curves: test RMSE (median of 30 tries)")
    print(f"   {'cars':>6}{'degree 2':>12}{'degree 9':>12}")
    for (n, _, te2), (_, _, te9) in zip(lc[2], lc[9]):
        print(f"   {n:>6,}{te2:>12,.0f}{te9:>12,.0f}")

    # ---------- charts ----------
    ages = np.linspace(0, 22, 300)

    # a. Three fits on the 12 cars
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, d, title in zip(axes, (0, 1, 11), ("degree 0: underfitting", "degree 1: just right", "degree 11: overfitting")):
        w = small_rows[d][3]
        ax.scatter(x[rest], y[rest], s=3, alpha=0.08, color="gray", label="test cars")
        ax.plot(ages, predict_poly(ages, w, d), color="#0E7A6B", lw=2.5)
        ax.scatter(x[small], y[small], color="black", zorder=3, label="12 training cars")
        ax.set(title=f"{title}\ntrain {small_rows[d][1]:,.0f} | test {small_rows[d][2]:,.0f} JOD",
               xlabel="car age (years)", ylim=(0, 40000))
    axes[0].set_ylabel("price (JOD)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(IMAGES / "three_fits.png", dpi=120)

    # b. Train vs test error by degree (12 cars)
    fig, ax = plt.subplots(figsize=(8, 4))
    degs = [r[0] for r in small_rows]
    ax.plot(degs, [r[1] for r in small_rows], "o-", label="train error (12 cars)")
    ax.plot(degs, [r[2] for r in small_rows], "o-", color="crimson", label="test error (other cars)")
    ax.set(xlabel="degree", ylabel="RMSE (JOD)", yscale="log",
           title="More complex: the train error falls, the test error explodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "error_by_degree.png", dpi=120)

    # c. CV by degree on all cars
    fig, ax = plt.subplots(figsize=(8, 4))
    ds = list(cv)
    ax.errorbar(ds, [cv[d][0] for d in ds], yerr=[cv[d][1] for d in ds], fmt="o-", capsize=4, label="CV error")
    ax.plot(ds, [cv[d][2] for d in ds], "s--", color="gray", label="train error")
    ax.axvline(simple, color="#0E7A6B", ls=":", label=f"chosen degree {simple}")
    ax.set(xlabel="degree", ylabel="RMSE (JOD)", title=f"All {len(df):,} cars: choosing the degree with cross-validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "cv_by_degree.png", dpi=120)

    # d. Learning curves
    fig, ax = plt.subplots(figsize=(8, 4))
    for d, color in ((2, "#0E7A6B"), (9, "#B7791F")):
        ax.plot([r[0] for r in lc[d]], [r[2] for r in lc[d]], "o-", color=color, label=f"degree {d}: test")
        ax.plot([r[0] for r in lc[d]], [r[1] for r in lc[d]], "o--", color=color, alpha=0.5, label=f"degree {d}: train")
    ax.set(xscale="log", yscale="log", xlabel="number of training cars", ylabel="RMSE (JOD)",
           title="Learning curves: more data tames the complex model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "learning_curves.png", dpi=120)
    print("\nCharts saved in images/")


if __name__ == "__main__":
    main()
