# -*- coding: utf-8 -*-
"""
ИСПОЛЬЗОВАНИЕ ГОТОВОЙ СУРРОГАТНОЙ МОДЕЛИ (surrogate_heater_model.pkl).

HYSYS здесь уже НЕ НУЖЕН — всё работает на сохранённой модели.

В скрипте два режима:
  A) ИНТЕРАКТИВНЫЙ — вводишь T_in, Flow_in, T_out вручную в консоли,
     сразу получаешь Q. Удобно, чтобы "поиграться" и проверить интуицию.
  B) ПАКЕТНЫЙ (what-if / сенситивити-анализ) — строит график того, как Q
     зависит от T_out при нескольких разных T_in, при фиксированном расходе.
     Это ровно то, для чего нужны суррогаты на практике: вместо того чтобы
     гонять HYSYS сотни раз, ты строишь целое "облако" сценариев за секунды.

Переключи режим ниже в MODE.
"""

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_FILE = "surrogate_heater_model.pkl"
MODE = "BATCH"   # "INTERACTIVE" или "BATCH"

# ---------------------------------------------------------------------------
# Загрузка модели (один раз)
# ---------------------------------------------------------------------------
bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
feature_cols = bundle["feature_cols"]  # [..., "Dist_to_boiling_C"]
t_sat_c = bundle.get("t_sat_c", 100.0)  # температура кипения, учтённая моделью


def predict_Q(t_in, flow_in, t_out):
    # признак "расстояние до кипения" считается автоматически из t_out —
    # вводить его вручную не нужно
    dist_to_boiling = t_out - t_sat_c
    x = np.array([[t_in, flow_in, t_out, dist_to_boiling]])
    return float(model.predict(x)[0])


# ---------------------------------------------------------------------------
# РЕЖИМ A: интерактивный ввод
# ---------------------------------------------------------------------------
def run_interactive():
    print("Интерактивный режим. Для выхода введи 'exit' вместо T_in.\n")
    while True:
        t_in_raw = input("T_in (C): ").strip()
        if t_in_raw.lower() == "exit":
            break
        try:
            t_in = float(t_in_raw)
            flow_in = float(input("Flow_in (kg/h): ").strip())
            t_out = float(input("T_out target (C): ").strip())
        except ValueError:
            print("  Нужно вводить числа. Попробуй ещё раз.\n")
            continue

        q = predict_Q(t_in, flow_in, t_out)
        print(f"  -> Q = {q:.1f} kJ/h\n")


# ---------------------------------------------------------------------------
# РЕЖИМ B: пакетный what-if / сенситивити-анализ
# ---------------------------------------------------------------------------
def run_batch():
    flow_in_fixed = 50.0                    # фиксируем расход
    t_in_values = [10.0, 20.0, 30.0, 40.0]  # несколько кривых — по одной на T_in
    t_out_range = np.linspace(60, 160, 50)  # развёртка по T_out

    plt.figure(figsize=(8, 6))
    for t_in in t_in_values:
        q_values = [predict_Q(t_in, flow_in_fixed, t_out) for t_out in t_out_range]
        plt.plot(t_out_range, q_values, marker="", label=f"T_in = {t_in:.0f} C")

    plt.xlabel("T_out (целевая температура выхода), C")
    plt.ylabel("Q, kJ/h (предсказание суррогата)")
    plt.title(f"Сенситивити-анализ: Q vs T_out при Flow_in = {flow_in_fixed:.0f} kg/h\n"
              f"(построено за секунды — без единого запуска HYSYS)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_file = "sensitivity_Q_vs_Tout.png"
    plt.savefig(out_file, dpi=150)
    print(f"График сохранён -> {out_file}")
    print(f"\nВсего посчитано точек: {len(t_in_values) * len(t_out_range)}")
    print("Если бы это считалось в HYSYS через COM — заняло бы заметное время")
    print("(каждая точка = пересчёт модели). Суррогат это сделал мгновенно.")


if __name__ == "__main__":
    if MODE == "INTERACTIVE":
        run_interactive()
    else:
        run_batch()
