# -*- coding: utf-8 -*-
"""
ФАЗА 8 — сравнение подходов на повторных кросс-валидациях + график неопределённости.

Один 5-fold CV с одним seed — это одно число; разбиение случайно, и разница между
моделями может оказаться шумом. Поэтому повторяем 5-fold CV N_REPEATS раз с разными
разбиениями и показываем разброс боксплотом.

Подходы (на равных: все "phys" получают q = Q/Flow и физические признаки):
  - Boosting (raw Q)           — бустинг на сырых признаках (как в Фазе 2)
  - Boosting (q + phys)        — бустинг на q и физических признаках
  - Phys. linear (q + phys)    — линейная физическая модель
  - GP (raw Q)                 — GP на сырых признаках
  - Enthalpy GP                — GP на q и физических признаках (развёрнутая модель)
  - Hybrid: linear + GP(resid) — физическая линейная модель + GP на её остатках

Вторая картинка: предсказание Q и полоса ±2σ по T_out, включая экстраполяцию
далеко за обучающий диапазон.

Запуск: python src/heater_08_compare_models.py  (HYSYS не нужен)
"""

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from heater_07_extrapolation import (RBFModel, SpecificModel, gb, plain_gp,
                                     quad_poly, small_nn)
from surrogate_model import EnthalpyGP, FEATURE_COLS, TARGET_COL, ROOT, get_domain

warnings.filterwarnings("ignore")

IN_XLSX = ROOT / "data" / "phase1_heater_dataset.xlsx"
OUT_BOX = ROOT / "figures" / "model_comparison_boxplot.png"
OUT_UNC = ROOT / "figures" / "uncertainty_band.png"
N_REPEATS = 10
N_FOLDS = 5
RANDOM_STATE = 42


class HybridModel(BaseEstimator, RegressorMixin):
    """Физическая линейная модель по q + GP на остатках q. Даёт и std (из GP)."""

    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        D = SpecificModel._design(X)
        q = y / X[:, 1]
        self.lin_ = LinearRegression().fit(D, q)
        resid = q - self.lin_.predict(D)
        kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0] * D.shape[1], nu=2.5) + WhiteKernel(1e-3)
        self.gp_ = make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=self.random_state))
        self.gp_.fit(D, resid)
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        D = SpecificModel._design(X)
        r, s = self.gp_[-1].predict(self.gp_[0].transform(D), return_std=True)
        mu = (self.lin_.predict(D) + r) * X[:, 1]
        return (mu, s * X[:, 1]) if return_std else mu


MODELS = {
    "Boosting (raw Q)":            lambda: gb(),
    "GP (raw Q)":                  lambda: plain_gp(),
    "Boosting (q + phys)":         lambda: SpecificModel(gb()),
    "Phys. linear (q + phys)":     lambda: SpecificModel(LinearRegression()),
    "Quad. poly (q + phys)":       lambda: SpecificModel(quad_poly()),
    "RBF (q + phys)":              lambda: SpecificModel(RBFModel()),
    "Small NN (q + phys)":         lambda: SpecificModel(small_nn()),
    "Enthalpy GP":                 lambda: EnthalpyGP(random_state=RANDOM_STATE),
    "Hybrid: linear + GP(resid)":  lambda: HybridModel(),
}

if __name__ == "__main__":
    df = pd.read_excel(IN_XLSX)
    df = df[df["Error"].isna() | (df["Error"] == "")]
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    if "--plots-only" not in sys.argv:   # --plots-only: только график неопределённости (без долгих CV)
        # --- повторные CV: метрики по каждому повтору (out-of-fold на всех точках) ---
        rkf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
        splits = list(rkf.split(X))
        results = {name: {"MAE": [], "R2": []} for name in MODELS}
        print(f"{N_REPEATS} повторов x {N_FOLDS}-fold CV, {len(MODELS)} моделей...")
        for rep in range(N_REPEATS):
            for name, make in MODELS.items():
                oof = np.zeros(len(y))
                for tr, te in splits[rep * N_FOLDS:(rep + 1) * N_FOLDS]:
                    oof[te] = make().fit(X[tr], y[tr]).predict(X[te])
                results[name]["MAE"].append(mean_absolute_error(y, oof))
                results[name]["R2"].append(r2_score(y, oof))
            print(f"  повтор {rep + 1}/{N_REPEATS} готов")

        print(f"\n{'Модель':30s} {'MAE медиана':>12s} {'MAE min..max':>22s} {'R2 медиана':>11s}")
        for name, r in results.items():
            m = np.array(r["MAE"])
            print(f"{name:30s} {np.median(m):12.1f} {m.min():10.1f} .. {m.max():9.1f} {np.median(r['R2']):11.5f}")

        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.boxplot([results[n]["MAE"] for n in MODELS], orientation="horizontal", tick_labels=list(MODELS))
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(f"MAE (out-of-fold, kJ/h), лог. шкала; боксплот по {N_REPEATS} повторам {N_FOLDS}-fold CV")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        OUT_BOX.parent.mkdir(exist_ok=True)
        plt.savefig(OUT_BOX, dpi=150)
        print(f"\nГрафик сохранён -> {OUT_BOX}")


    # --- график неопределённости (в удельной величине q = Q/Flow: расход выпадает) ---
    dom = get_domain()
    t_in0, flow0 = 25.0, 55.0
    t_grid = np.linspace(30, 220, 400)
    Xg = np.column_stack([np.full_like(t_grid, t_in0), np.full_like(t_grid, flow0), t_grid])
    full = {n: MODELS[n]().fit(X, y) for n in ("Enthalpy GP", "Hybrid: linear + GP(resid)")}
    gp_raw = plain_gp().fit(X, y)

    curves = {}   # имя -> (q, sigma_q, цвет)
    for name, mdl, color in [("Enthalpy GP", full["Enthalpy GP"], "tab:green"),
                             ("Hybrid", full["Hybrid: linear + GP(resid)"], "tab:blue")]:
        mu, sd = mdl.predict(Xg, return_std=True)
        curves[name] = (mu / flow0, sd / flow0, color)
    mu, sd = gp_raw[-1].predict(gp_raw[0].transform(Xg), return_std=True)
    curves["GP (raw Q)"] = (mu / flow0, sd / flow0, "tab:red")

    near = np.abs(df.T_in_C - t_in0) < 8                       # q слабо зависит от T_in (~1 %)
    q_pts = df.Q_kJ_per_h / df.Flow_in_kgh
    lo, hi = dom["T_out_target_C"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for k, ax in enumerate(axes[:2]):
        for name, (q, sq, color) in curves.items():
            ax.plot(t_grid, q, color=color, ls="--" if name.startswith("GP") else "-", label=f"{name}: q")
            ax.fill_between(t_grid, q - 2 * sq, q + 2 * sq, color=color, alpha=0.2)
        ax.scatter(df.T_out_target_C[near], q_pts[near], s=14, color="k", zorder=5,
                   label="обучающие точки (|T_in-25|<8)")
        ax.axvline(100, color="gray", ls=":")
        ax.set_xlabel("T_out, C")
        ax.set_ylabel("q = Q/Flow, kJ/kg")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.axvspan(t_grid[0], lo, color="gray", alpha=0.12)
            ax.axvspan(hi, t_grid[-1], color="gray", alpha=0.12, label="вне обучающего диапазона")
            ax.set_ylim(-1500, 6000)
            ax.set_title(f"q(T_out) и ±2σ (T_in={t_in0:g}); серым — вне диапазона обучения")
            ax.legend(fontsize=8, loc="upper left")
        else:
            ax.set_xlim(90, 110)
            ax.set_ylim(0, 4500)
            ax.set_title("Увеличено у точки кипения")
    ax = axes[2]
    for name, (q, sq, color) in curves.items():
        ax.plot(t_grid, 2 * sq * flow0, color=color, ls="--" if name.startswith("GP") else "-", label=name)
    ax.axvspan(t_grid[0], lo, color="gray", alpha=0.12)
    ax.axvspan(hi, t_grid[-1], color="gray", alpha=0.12)
    ax.axvline(100, color="gray", ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("T_out, C")
    ax.set_ylabel("2σ по Q (при Flow = 55), kJ/h, лог.")
    ax.set_title("Предсказанная неопределённость")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_UNC, dpi=150)
    print(f"График сохранён -> {OUT_UNC}")

    # --- численно: σ внутри домена и далеко снаружи ---
    mu, sd = full["Enthalpy GP"].predict(
        np.array([[t_in0, flow0, t] for t in (60, 100.3, 140, 159, 180, 220, 300)]), return_std=True)
    print("\nEnthalpy GP σ по T_out:", dict(zip((60, 100.3, 140, 159, 180, 220, 300), np.round(sd).astype(int).tolist())))
