# -*- coding: utf-8 -*-
"""
ФАЗА 1 — генерация датасета методом Latin Hypercube Sampling (LHS).

Модель: Water In -> E-100 (Heater) -> Water Out, поток энергии Q-100.

В отличие от Фазы 0 (где менялась только 1 переменная), здесь варьируются
СРАЗУ НЕСКОЛЬКО входов одновременно:
  - температура Water In
  - массовый расход Water In
  - целевая температура Water Out (задание для нагревателя)

LHS даёт более равномерное покрытие пространства входов, чем случайная
выборка или перебор по сетке — это стандартный метод Design of Experiments (DoE)
для суррогатного моделирования.

Скрипт:
  1. Подключается к уже запущенному HYSYS.
  2. Генерирует N_SAMPLES точек плана эксперимента (LHS, без внешних библиотек).
  3. Для каждой точки задаёт входы, читает отклик модели (Q, доля пара и т.д.).
  4. Сохраняет полный датасет в CSV, с колонкой Error для точек, где HYSYS
     не смог сойтись или вернул ошибку (такие точки не теряются молча,
     а остаются в таблице — это удобно для диагностики).
"""

import win32com.client as win32
import csv
import random

try:
    import openpyxl
    from openpyxl.styles import Font
    HAVE_OPENPYXL = True
except ImportError:
    HAVE_OPENPYXL = False

# ---------------------------------------------------------------------------
# НАСТРОЙКИ — поменяй под свои имена и диапазоны
# ---------------------------------------------------------------------------
STREAM_IN   = "Water In"
STREAM_OUT  = "Water Out"
ENERGY_NAME = "Q-100"

# Диапазоны входных переменных (мин, макс)
T_IN_RANGE    = (10.0, 40.0)    # температура Water In, C
FLOW_IN_RANGE = (30.0, 80.0)    # массовый расход Water In, kg/h
T_OUT_RANGE   = (60.0, 160.0)   # целевая температура Water Out, C

N_SAMPLES = 150                 # сколько точек плана эксперимента сгенерировать
                                 # (подняли с 50 до 150 — для более надёжной
                                 # оценки качества модели в Фазе 2)
RANDOM_SEED = 42                # для воспроизводимости; поставь None для случайного плана
OUT_XLSX = "phase1_dataset.xlsx"  # основной результат — нативный Excel-файл
OUT_CSV = "phase1_dataset.csv"    # резервная копия (на случай если openpyxl недоступен)


# ---------------------------------------------------------------------------
# LHS без numpy/scipy — чистый Python, чтобы не тащить лишние зависимости
# ---------------------------------------------------------------------------
def latin_hypercube_sampling(n_samples, ranges, seed=None):
    """
    ranges: список кортежей (min, max), по одному на переменную.
    Возвращает список строк (каждая строка — список значений, в порядке ranges).

    Идея LHS: каждую переменную делим на n_samples равных интервалов (страт),
    в каждом интервале берём 1 случайную точку, затем независимо перемешиваем
    порядок страт для каждой переменной. Это гарантирует, что при проекции
    на любую ось все n_samples интервалов использованы ровно по разу.
    """
    rng = random.Random(seed)
    n_dims = len(ranges)
    result = [[0.0] * n_dims for _ in range(n_samples)]

    for dim, (lo, hi) in enumerate(ranges):
        # границы страт
        strata = list(range(n_samples))
        rng.shuffle(strata)  # порядок страт для этой переменной — независимый
        width = (hi - lo) / n_samples
        for row in range(n_samples):
            stratum = strata[row]
            # случайная точка внутри страты [stratum*width, (stratum+1)*width]
            offset = rng.uniform(0.0, width)
            value = lo + stratum * width + offset
            result[row][dim] = value

    return result


# ---------------------------------------------------------------------------
# ШАГ 1. Подключение к запущенному HYSYS
# ---------------------------------------------------------------------------
print("Подключаюсь к HYSYS...")
hysys = win32.GetObject(Class="HYSYS.Application")
case = hysys.ActiveDocument
print(f"  OK. Открытый кейс: {case.Title.Value}")

flowsheet = case.Flowsheet
streams = flowsheet.MaterialStreams
energy = flowsheet.EnergyStreams

s_in = streams.Item(STREAM_IN)
s_out = streams.Item(STREAM_OUT)
q = energy.Item(ENERGY_NAME)

# ---------------------------------------------------------------------------
# ШАГ 2. Генерация плана эксперимента (LHS)
# ---------------------------------------------------------------------------
print(f"\nГенерирую LHS-план: {N_SAMPLES} точек, 3 переменные...")
design = latin_hypercube_sampling(
    N_SAMPLES,
    [T_IN_RANGE, FLOW_IN_RANGE, T_OUT_RANGE],
    seed=RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# ШАГ 3. Прогон плана через HYSYS
# ---------------------------------------------------------------------------
rows = []
print("\nЗапускаю прогон датасета...")
for i, (t_in, flow_in, t_out) in enumerate(design, start=1):
    row = {
        "sample": i,
        "T_in_C": round(t_in, 2),
        "Flow_in_kgh": round(flow_in, 2),
        "T_out_target_C": round(t_out, 2),
        "Q_kJ_per_h": None,
        "VapFrac_out": None,
        "T_out_actual_C": None,
        "Error": "",
    }
    try:
        # задаём входы
        s_in.Temperature.SetValue(t_in, "C")
        s_in.MassFlow.SetValue(flow_in, "kg/h")
        s_out.Temperature.SetValue(t_out, "C")

        # читаем отклик модели
        q_val = q.HeatFlow.GetValue("kJ/h")
        vapfrac = s_out.VapourFraction.GetValue()
        t_out_actual = s_out.Temperature.GetValue("C")

        row["Q_kJ_per_h"] = round(q_val, 1)
        row["VapFrac_out"] = round(vapfrac, 4)
        row["T_out_actual_C"] = round(t_out_actual, 2)

        print(f"  [{i:3d}/{N_SAMPLES}] T_in={t_in:6.2f}  Flow_in={flow_in:6.2f}  "
              f"T_out={t_out:6.2f}  ->  Q={q_val:12.1f} kJ/h  (пар={vapfrac:.3f})")

    except Exception as e:
        # точка не потеряна — попадёт в CSV с пометкой об ошибке
        row["Error"] = str(e)
        print(f"  [{i:3d}/{N_SAMPLES}] ОШИБКА на точке T_in={t_in:.2f}, "
              f"Flow_in={flow_in:.2f}, T_out={t_out:.2f}: {e}")

    rows.append(row)

# ---------------------------------------------------------------------------
# ШАГ 4. Сохранить датасет
# ---------------------------------------------------------------------------
fieldnames = ["sample", "T_in_C", "Flow_in_kgh", "T_out_target_C",
              "Q_kJ_per_h", "VapFrac_out", "T_out_actual_C", "Error"]

saved_files = []

if HAVE_OPENPYXL:
    # Основной путь: настоящий .xlsx с числовыми ячейками.
    # Это полностью обходит проблему с региональными настройками Excel
    # (в русской локали разделитель CSV — ";", а не ",", из-за чего
    # обычный .csv разъезжается в одну колонку при открытии двойным кликом).
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "phase1_dataset"

    ws.append(fieldnames)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for row in rows:
        ws.append([row[col] for col in fieldnames])

    # автоширина колонок (по содержимому)
    for col_idx, col_name in enumerate(fieldnames, start=1):
        letter = ws.cell(row=1, column=col_idx).column_letter
        width = max(len(col_name), 10)
        ws.column_dimensions[letter].width = width + 2

    ws.freeze_panes = "A2"
    wb.save(OUT_XLSX)
    saved_files.append(OUT_XLSX)
else:
    print("\n[!] Модуль openpyxl не установлен — .xlsx не создан.")
    print("    Установи его командой:  py -m pip install openpyxl")
    print("    и перезапусти скрипт для получения нормального Excel-файла.")

# CSV всё равно сохраняем как резервную копию — с ";" (разделитель для RU-локали)
with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(rows)
saved_files.append(OUT_CSV)

n_ok = sum(1 for r in rows if not r["Error"])
n_err = len(rows) - n_ok
print(f"\nГотово. {n_ok} точек успешно, {n_err} с ошибкой -> {', '.join(saved_files)}")
print("Этот датасет уже пригоден для обучения первой суррогатной модели")
print("(например, линейная регрессия или Gradient Boosting: вход T_in/Flow_in/T_out -> выход Q).")
