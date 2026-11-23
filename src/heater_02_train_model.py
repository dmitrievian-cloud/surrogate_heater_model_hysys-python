# -*- coding: utf-8 -*-
"""
ФАЗА 2 (v2) — обучение суррогатной модели с КРОСС-ВАЛИДАЦИЕЙ.

Почему переделали относительно первой версии:
  В первой версии метрики (R^2, MAE) считались на одном случайном
  test-наборе из 10 точек — это мало, и одна "неудачная" точка могла
  сильно исказить оценку качества модели.

Что изменилось:
  Вместо одного train/test split используется K-Fold кросс-валидация
  (K=5): датасет делится на 5 частей, модель 5 раз обучается на 4/5
  данных и проверяется на оставшейся 1/5. В итоге у КАЖДОЙ точки
  датасета появляется "честное" предсказание — сделанное моделью,
  которая эту точку не видела при обучении (out-of-fold prediction).

  Это даёт:
    - метрику R^2/MAE, усреднённую по 5 независимым проверкам
      (+ разброс std — насколько стабильна оценка)
    - parity-график на ВСЕХ точках датасета, а не только на 20%

Что делает скрипт:
  1. Читает phase1_dataset.xlsx.
  2. Признаки: T_in_C, Flow_in_kgh, T_out_target_C. (Раньше был ещё
     Dist_to_boiling_C = T_out - 100, но он убран: это тот же T_out, только
     сдвинутый на константу. Деревья Gradient Boosting режут по порогам и
     сдвиг не замечают, а у линейной регрессии такой столбец полностью
     повторяет T_out. Новой информации он не давал; метрики не изменились.)
  3. Прогоняет 5-fold кросс-валидацию для Linear Regression и
     Gradient Boosting, собирая out-of-fold предсказания.
  4. Печатает R^2/MAE по фолдам (среднее ± std).
  5. Строит parity-график на всех точках (честные out-of-fold предсказания).
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from surrogate_model import EnthalpyGP, QuadPolyModel   # GP по q = Q/Flow + флаг фазы (см. surrogate_model.py)

# ---------------------------------------------------------------------------
# НАСТРОЙКИ
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent   # корень репозитория (src/ лежит на уровень ниже)
IN_XLSX = ROOT / "data" / "phase1_heater_dataset.xlsx"
FEATURE_COLS = ["T_in_C", "Flow_in_kgh", "T_out_target_C"]
TARGET_COL = "Q_kJ_per_h"
N_FOLDS = 5
RANDOM_STATE = 42
OUT_PLOT = ROOT / "figures" / "phase2_parity_plot.png"

# ---------------------------------------------------------------------------
# ШАГ 1. Загрузка и очистка данных
# ---------------------------------------------------------------------------
print(f"Читаю {IN_XLSX}...")
df = pd.read_excel(IN_XLSX)

n_total = len(df)
df = df[df["Error"].isna() | (df["Error"] == "")]
n_clean = len(df)
print(f"  Всего точек: {n_total}, годных (без ошибок): {n_clean}")

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# ШАГ 2. Кросс-валидация для обеих моделей
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
    ),
    # GP: отдельный масштаб длины по каждому признаку (ARD), шум подбирается сам.
    # Даёт не только предсказание, но и std (см. predict(..., return_std=True)).
    "Gaussian Process": make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=[1.0] * len(FEATURE_COLS), nu=2.5)
                   + WhiteKernel(1e-3),
            normalize_y=True, n_restarts_optimizer=5, random_state=RANDOM_STATE,
        ),
    ),
    # GP на удельной величине q = Q/Flow + флаг фазы: скачок при кипении уходит в флаг
    "Enthalpy GP": EnthalpyGP(random_state=RANDOM_STATE),
    # квадратичный полином по q + флаг фазы — РАЗВЁРНУТАЯ модель (см. heater_03)
    "Quadratic poly (q + phase flag)": QuadPolyModel(random_state=RANDOM_STATE),
}

results = {}
print(f"\nЗапускаю {N_FOLDS}-fold кросс-валидацию...")
for name, model in models.items():
    # метрики по фолдам (для среднего и разброса)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")

    # честные предсказания "каждая точка предсказана моделью, её не видевшей"
    y_pred_oof = cross_val_predict(model, X, y, cv=kf)

    r2_overall = r2_score(y, y_pred_oof)
    mae_overall = mean_absolute_error(y, y_pred_oof)

    results[name] = {
        "y_pred_oof": y_pred_oof,
        "r2_folds": r2_scores,
        "mae_folds": mae_scores,
        "r2_overall": r2_overall,
        "mae_overall": mae_overall,
    }

    print(f"\n  {name}")
    print(f"    R^2 по фолдам: {np.round(r2_scores, 4)}")
    print(f"    R^2 = {r2_scores.mean():.4f} +/- {r2_scores.std():.4f}  (среднее +/- std)")
    print(f"    MAE по фолдам: {np.round(mae_scores, 1)}")
    print(f"    MAE = {mae_scores.mean():.1f} +/- {mae_scores.std():.1f} kJ/h")
    print(f"    R^2/MAE на всех out-of-fold предсказаниях: "
          f"R^2={r2_overall:.4f}, MAE={mae_overall:.1f} kJ/h")

# ---------------------------------------------------------------------------
# ШАГ 3. Parity plot на всех точках (out-of-fold)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

for ax, (name, res) in zip(axes.ravel(), results.items()):
    y_pred = res["y_pred_oof"]
    ax.scatter(y, y_pred, alpha=0.6, edgecolor="k", s=30)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="идеальное совпадение")
    ax.set_xlabel("Факт (HYSYS), kJ/h")
    ax.set_ylabel("Предсказание (out-of-fold), kJ/h")
    ax.set_title(f"{name}\nR² = {res['r2_overall']:.4f}, MAE = {res['mae_overall']:.0f} kJ/h\n"
                 f"(5-fold CV, n={len(y)})")
    ax.legend()
    ax.grid(alpha=0.3)

for ax in axes.ravel()[len(results):]:   # лишние панели сетки прячем
    ax.axis("off")
plt.tight_layout()
OUT_PLOT.parent.mkdir(exist_ok=True)
plt.savefig(OUT_PLOT, dpi=150)
print(f"\nГрафик сохранён -> {OUT_PLOT}")

# ---------------------------------------------------------------------------
# ШАГ 4. Итог
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda n: results[n]["r2_overall"])
print(f"\nЛучшая модель по R^2 (кросс-валидация, честная оценка): {best_name}")
print("\nПочему это надёжнее предыдущей версии:")
print("  - метрика считается на ВСЕХ точках датасета, а не на случайных 20%")
print("  - разброс (std) по фолдам показывает, насколько стабильна оценка:")
print("    маленький std -> модели можно доверять, большой std -> нужно больше данных")
