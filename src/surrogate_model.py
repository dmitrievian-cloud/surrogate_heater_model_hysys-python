# -*- coding: utf-8 -*-
"""
Общий код суррогата нагревателя: модель и функция predict_Q.

Идея ("энтальпийная" постановка):
  Q = m * (h_out - h_in), то есть удельная величина q = Q / Flow зависит
  только от температур, а расход выпадает. Поэтому GP учится не на Q, а на
  q = Q / Flow_in (kJ/kg) по признакам [T_in, T_out, is_vapor], где
  is_vapor = (T_out >= T_sat) — флаг фазы на выходе (при ~101 kPa T_sat ~ 100 C).
  Скачок при кипении (скрытая теплота) целиком уходит в этот флаг, а гладкая
  часть q(T) остаётся для GP. Итоговое Q = q * Flow_in; std масштабируется так же.

  Это НЕ то же самое, что удалённый ранее Dist_to_boiling_C = T_out - 100:
  тот был сдвигом на константу, а флаг фазы — ступенька, новая информация.

Ограничение: T_sat зашита как константа (давление в кейсе фиксировано).
При другом давлении её нужно пересчитать и переобучить модель.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "surrogate_heater_model.pkl"
FEATURE_COLS = ["T_in_C", "Flow_in_kgh", "T_out_target_C"]   # входы predict(): порядок важен
TARGET_COL = "Q_kJ_per_h"
T_SAT_C = 100.0


class EnthalpyGP(BaseEstimator, RegressorMixin):
    """GP по q = Q/Flow; вход X = [T_in, Flow_in, T_out], выход — Q (kJ/h)."""

    def __init__(self, t_sat_c=T_SAT_C, random_state=42):
        self.t_sat_c = t_sat_c
        self.random_state = random_state

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        t_in, t_out = X[:, 0], X[:, 2]
        return np.c_[t_in, t_out, (t_out >= self.t_sat_c).astype(float)]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        q = np.asarray(y, dtype=float) / X[:, 1]
        kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0] * 3, nu=2.5) + WhiteKernel(1e-3)
        self.pipe_ = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                     n_restarts_optimizer=5, random_state=self.random_state),
        )
        self.pipe_.fit(self._design(X), q)
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        D = self.pipe_[0].transform(self._design(X))
        mu, sd = self.pipe_[-1].predict(D, return_std=True)
        flow = X[:, 1]
        return (mu * flow, sd * flow) if return_std else mu * flow


_BUNDLE = None


def load_model(model_file=MODEL_FILE):
    """Файл модели читается с диска ОДИН раз, дальше берётся из памяти."""
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(model_file)
    return _BUNDLE["model"]


def predict_Q(t_in, flow_in, t_out, return_std=False):
    """Q (kJ/h) по T_in (C), Flow_in (kg/h), T_out (C); опционально ещё и std (kJ/h)."""
    model = load_model()
    x = [[t_in, flow_in, t_out]]
    if return_std:
        mu, sd = model.predict(x, return_std=True)
        return float(mu[0]), float(sd[0])
    return float(model.predict(x)[0])
