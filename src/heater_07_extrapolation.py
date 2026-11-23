# -*- coding: utf-8 -*-
"""
ФАЗА 7 — честная проверка экстраполяции.

Случайная k-fold CV проверяет ИНТЕРПОЛЯЦИЮ: тестовые точки лежат среди обучающих.
Здесь модель обучается на ЧАСТИ диапазона входов и проверяется на оставшейся
части, которую она не видела:
  1. Flow:  обучение 30-60 kg/h  ->  тест 60-80 kg/h
  2. T_in:  обучение T_in <= 30  ->  тест T_in > 30
  3. T_out: обучение T_out <= 130 -> тест T_out > 130 (дальше в область пара)

Модели сравниваются на РАВНЫХ условиях: и бустинг, и линейная модель получают
те же физические признаки и целевую q = Q/Flow, что и Enthalpy GP — иначе
сравнивались бы одновременно класс модели и инженерия признаков.

Запуск: python src/heater_07_extrapolation.py  (HYSYS не нужен)
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from surrogate_model import EnthalpyGP, FEATURE_COLS, T_SAT_C, TARGET_COL, ROOT

warnings.filterwarnings("ignore")

IN_XLSX = ROOT / "data" / "phase1_heater_dataset.xlsx"
OUT_PLOT = ROOT / "figures" / "extrapolation_test.png"
RANDOM_STATE = 42


class SpecificModel(BaseEstimator, RegressorMixin):
    """Любой регрессор base на q = Q/Flow с физическими признаками [T_in, T_out, is_vapor, max(0,T_out-100)]."""

    def __init__(self, base=None):
        self.base = base

    @staticmethod
    def _design(X):
        t_in, t_out = X[:, 0], X[:, 2]
        return np.column_stack([t_in, t_out, (t_out >= T_SAT_C).astype(float),
                                np.maximum(0.0, t_out - T_SAT_C)])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.base_ = clone(self.base).fit(self._design(X), y / X[:, 1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.base_.predict(self._design(X)) * X[:, 1]


def plain_gp():
    kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0] * 3, nu=2.5) + WhiteKernel(1e-3)
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=RANDOM_STATE))


def gb():
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                                     random_state=RANDOM_STATE)


MODELS = {
    "Linear (raw Q)":            lambda: LinearRegression(),
    "Boosting (raw Q)":          lambda: gb(),
    "GP (raw Q)":                lambda: plain_gp(),
    "Boosting (q + phys)":       lambda: SpecificModel(gb()),
    "Phys. linear (q + phys)":   lambda: SpecificModel(LinearRegression()),
    "Enthalpy GP":               lambda: EnthalpyGP(random_state=RANDOM_STATE),
    "Enthalpy GP + kink":        lambda: EnthalpyGP(use_kink=True, random_state=RANDOM_STATE),
}

SPLITS = {   # имя: (колонка, функция "это обучающая точка")
    "Flow 30-60 -> 60-80":  ("Flow_in_kgh",    lambda v: v <= 60),
    "T_in <=30 -> >30":     ("T_in_C",         lambda v: v <= 30),
    "T_out <=130 -> >130":  ("T_out_target_C", lambda v: v <= 130),
}


def predict_std(model, X):
    """(mean, std) для GP-моделей; (mean, None) для остальных."""
    if isinstance(model, EnthalpyGP):
        return model.predict(X, return_std=True)
    if hasattr(model, "__getitem__") and isinstance(model[-1], GaussianProcessRegressor):
        mu, sd = model[-1].predict(model[0].transform(X), return_std=True)
        return mu, sd
    return model.predict(X), None


if __name__ == "__main__":
    df = pd.read_excel(IN_XLSX)
    df = df[df["Error"].isna() | (df["Error"] == "")]
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    rows, details = [], {}
    for split_name, (col, is_train) in SPLITS.items():
        tr = is_train(df[col].values)
        te = ~tr
        print(f"\n=== {split_name}: обучение {tr.sum()} точек, тест {te.sum()} точек ===")
        for mname, make in MODELS.items():
            model = make().fit(X[tr], y[tr])
            mu, sd = predict_std(model, X[te])
            err = np.abs(y[te] - mu)
            row = {"split": split_name, "model": mname,
                   "R2": r2_score(y[te], mu), "MAE": mean_absolute_error(y[te], mu),
                   "mean_sigma": np.nan if sd is None else sd.mean(),
                   "in_2sigma_%": np.nan if sd is None else 100 * (err <= 2 * sd).mean()}
            rows.append(row)
            details[(split_name, mname)] = (X[te], err, sd)
            sig = "" if sd is None else f"  sigma={sd.mean():9.0f}  |err|<=2sigma: {row['in_2sigma_%']:3.0f}%"
            print(f"  {mname:24s} R2={row['R2']:9.4f}  MAE={row['MAE']:11.1f} kJ/h{sig}")

    res = pd.DataFrame(rows)

    # --- график: MAE по моделям и сплитам (лог-шкала) + ошибка vs 2*sigma для Enthalpy GP ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for j, (split_name, (col, _)) in enumerate(SPLITS.items()):
        ax = axes[0, j]
        r = res[res.split == split_name]
        ax.barh(r.model, r.MAE, color=["#999"] * 3 + ["#7aa6d6"] * 2 + ["#2a8f3c"] * 2)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_title(split_name)
        ax.set_xlabel("MAE на непросмотренной части, kJ/h (лог.)")
        ax.grid(alpha=0.3, axis="x")

        Xte, err, sd = details[(split_name, "Enthalpy GP")]
        ax = axes[1, j]
        order = np.argsort(Xte[:, FEATURE_COLS.index(col)])
        xv = Xte[order, FEATURE_COLS.index(col)]
        ax.plot(xv, 2 * sd[order], "r-", label="2σ (предсказанная)")
        ax.scatter(xv, err[order], s=18, color="k", label="|ошибка| факт")
        ax.set_yscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("kJ/h (лог.)")
        ax.set_title("Enthalpy GP: ошибка vs 2σ на тесте")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    OUT_PLOT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"\nГрафик сохранён -> {OUT_PLOT}")
