# -*- coding: utf-8 -*-
"""
ФАЗА 6 — оптимизация поверх суррогата (what-if для принятия решений).

Суррогат отвечает за доли миллисекунды, поэтому его можно "крутить" в
оптимизаторе тысячи раз — с HYSYS это было бы слишком медленно.

Ограничение по мощности делаем КОНСЕРВАТИВНЫМ: требуем Q + K*sigma <= Q_max,
где sigma — предсказанная неопределённость GP, а K = SIGMA_K (по умолчанию 2).
Так оптимизатор не "втискивается" в запас, который модель знает не точно.

Две задачи (обе по одной переменной, Q монотонно растёт по обеим):
  A. Максимальный расход Flow при заданных T_in, T_out и лимите нагревателя Q_max.
  B. Максимальная достижимая T_out при заданных T_in, Flow и лимите Q_max.
     (На T_sat Q скачком растёт на скрытую теплоту: если лимит попадает в этот
     скачок, нагреватель упирается в испарение — скрипт это сообщает.)

HYSYS здесь НЕ нужен.
"""

import warnings

from scipy.optimize import brentq

from surrogate_model import get_domain, predict_Q, T_SAT_C

SIGMA_K = 2.0   # сколько sigma запаса закладываем в ограничение по Q


def q_upper(t_in, flow_in, t_out):
    """Консервативная оценка Q: среднее + K*sigma."""
    q, std = predict_Q(t_in, flow_in, t_out, return_std=True)
    return q + SIGMA_K * std


def max_flow(t_in, t_out, q_max):
    """Задача A: максимальный расход (в диапазоне обучения) при Q_upper <= q_max."""
    lo, hi = get_domain()["Flow_in_kgh"]
    g = lambda f: q_upper(t_in, f, t_out) - q_max
    if g(lo) > 0:
        return None, "лимит Q_max слишком мал: нельзя нагреть даже минимальный расход"
    if g(hi) <= 0:
        return hi, "ограничение не активно: упёрлись в верхнюю границу расхода"
    return brentq(g, lo, hi, xtol=1e-4), "ограничение по Q активно"


def max_t_out(t_in, flow_in, q_max):
    """Задача B: максимальная T_out (в диапазоне обучения) при Q_upper <= q_max."""
    lo, hi = get_domain()["T_out_target_C"]
    g = lambda t: q_upper(t_in, flow_in, t) - q_max
    # Q(T_out) имеет скачок на T_sat (скрытая теплота), поэтому жидкую (T < T_sat)
    # и паровую (T >= T_sat) ветви ищем раздельно: brentq на разрыве мог бы
    # вернуть точку с паровой стороны, где лимит уже нарушен.
    liq_hi = min(hi, T_SAT_C - 1e-6)
    if g(lo) > 0:
        return None, "лимит Q_max слишком мал даже для минимальной T_out"
    if g(liq_hi) > 0:                                   # корень на жидкой ветви
        return brentq(g, lo, liq_hi, xtol=1e-4), "ограничение по Q активно (жидкость)"
    if hi < T_SAT_C:                                    # паровой ветви в домене нет
        return hi, "ограничение не активно: упёрлись в верхнюю границу T_out"
    if g(T_SAT_C) > 0:                                  # лимит попал в скачок
        return liq_hi, (f"упёрлись в испарение при {T_SAT_C:g} C: лимита не хватает "
                        f"на скрытую теплоту, берём жидкость на границе")
    if g(hi) <= 0:
        return hi, "ограничение не активно: упёрлись в верхнюю границу T_out"
    return brentq(g, T_SAT_C, hi, xtol=1e-4), "ограничение по Q активно (пар)"


if __name__ == "__main__":
    warnings.simplefilter("always")

    print("=== Задача A: максимальный расход ===")
    t_in, t_out = 20.0, 130.0
    for q_max in (60_000, 120_000, 150_000, 200_000):
        f, note = max_flow(t_in, t_out, q_max)
        if f is None:
            print(f"  T_in={t_in:.0f} C, T_out={t_out:.0f} C, Q_max={q_max:>7,} kJ/h -> {note}")
        else:
            q, std = predict_Q(t_in, f, t_out, return_std=True)
            print(f"  T_in={t_in:.0f} C, T_out={t_out:.0f} C, Q_max={q_max:>7,} kJ/h -> "
                  f"Flow_max = {f:5.1f} kg/h  (Q = {q:,.0f} +/- {std:,.0f}; {note})")

    print("\n=== Задача B: максимальная T_out ===")
    t_in, flow = 20.0, 50.0
    for q_max in (10_000, 20_000, 60_000, 120_000, 140_000, 200_000):
        t, note = max_t_out(t_in, flow, q_max)
        if t is None:
            print(f"  T_in={t_in:.0f} C, Flow={flow:.0f} kg/h, Q_max={q_max:>7,} kJ/h -> {note}")
        else:
            q, std = predict_Q(t_in, flow, t, return_std=True)
            print(f"  T_in={t_in:.0f} C, Flow={flow:.0f} kg/h, Q_max={q_max:>7,} kJ/h -> "
                  f"T_out_max = {t:6.2f} C  (Q = {q:,.0f} +/- {std:,.0f}; {note})")
