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

import warnings
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

    def __init__(self, t_sat_c=T_SAT_C, use_kink=False, random_state=42):
        self.t_sat_c = t_sat_c
        self.use_kink = use_kink      # + признак max(0, T_out - T_sat): излом наклона q(T) в паре
        self.random_state = random_state

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        t_in, t_out = X[:, 0], X[:, 2]
        cols = [t_in, t_out, (t_out >= self.t_sat_c).astype(float)]
        if self.use_kink:
            cols.append(np.maximum(0.0, t_out - self.t_sat_c))
        return np.column_stack(cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        q = np.asarray(y, dtype=float) / X[:, 1]
        n_feat = 4 if self.use_kink else 3
        kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0] * n_feat, nu=2.5) + WhiteKernel(1e-3)
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


def get_domain():
    """Диапазоны обучения {имя признака: (min, max)} или None, если модель их не хранит."""
    load_model()
    return _BUNDLE.get("domain")


def check_domain(t_in, flow_in, t_out, warn=True):
    """Проверка входов (числа или массивы) на попадание в диапазон обучения.

    Возвращает булев массив «точка в домене»; по каждому признаку с нарушениями
    выдаёт ОДИН RuntimeWarning (с числом точек и крайним значением)."""
    t_in, flow_in, t_out = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (t_in, flow_in, t_out)))
    ok = np.ones(t_in.shape, dtype=bool)
    domain = get_domain()   # {имя признака: (min, max)}, сохраняется в heater_03
    if not domain:
        return ok
    for name, vals in zip(FEATURE_COLS, (t_in, flow_in, t_out)):
        lo, hi = domain[name]
        bad = (vals < lo) | (vals > hi)
        if bad.any():
            ok &= ~bad
            if not warn:
                continue
            far = vals[bad][np.argmax(np.abs(vals[bad] - np.clip(vals[bad], lo, hi)))]
            warnings.warn(f"{name}: {int(bad.sum())} знач. вне диапазона обучения [{lo:g}, {hi:g}] "
                          f"(крайнее: {far:g}) — предсказание ненадёжно", RuntimeWarning, stacklevel=3)
    return ok


def predict_Q(t_in, flow_in, t_out, return_std=False):
    """Q (kJ/h) по T_in (C), Flow_in (kg/h), T_out (C); опционально ещё и std (kJ/h).

    Аргументы — числа ИЛИ массивы (вещательное сложение, как в numpy): для чисел
    возвращается float, для массивов — массив формы вещания. Вне диапазона обучения
    выдаётся RuntimeWarning."""
    model = load_model()
    check_domain(t_in, flow_in, t_out)
    a, b, c = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (t_in, flow_in, t_out)))
    x = np.column_stack([a.ravel(), b.ravel(), c.ravel()])
    if return_std:
        mu, sd = model.predict(x, return_std=True)
        mu, sd = mu.reshape(a.shape), sd.reshape(a.shape)
        return (float(mu), float(sd)) if a.ndim == 0 else (mu, sd)
    mu = model.predict(x).reshape(a.shape)
    return float(mu) if a.ndim == 0 else mu


def predict_interval(t_in, flow_in, t_out, k=2.0):
    """Консервативный коридор (mean - k*sigma, mean + k*sigma) для Q, kJ/h.

    Для ограничений "не больше" берите верхнюю границу, для "не меньше" — нижнюю."""
    mu, sd = predict_Q(t_in, flow_in, t_out, return_std=True)
    return mu - k * sd, mu + k * sd


def is_trusted(t_in, flow_in, t_out):
    """Флаг доверия (булев, числа или массив): вход внутри диапазона обучения И
    удельная неопределённость sigma/Flow не выше порога, посчитанного при обучении
    (1.25 x максимум out-of-fold sigma/Flow, см. heater_03). Без предупреждений."""
    model = load_model()
    ok = check_domain(t_in, flow_in, t_out, warn=False)
    thr = _BUNDLE.get("sigma_q_trust")
    a, b, c = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (t_in, flow_in, t_out)))
    if thr is not None:
        _, sd = model.predict(np.column_stack([a.ravel(), b.ravel(), c.ravel()]), return_std=True)
        ok = ok & (sd.reshape(a.shape) / b <= thr)
    return bool(ok) if a.ndim == 0 else ok
