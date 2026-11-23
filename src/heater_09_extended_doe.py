# -*- coding: utf-8 -*-
"""
ФАЗА 9 — расширенный DoE: дополнительные точки для суррогата.

Зачем:
  1. У фазового перехода (100 °C) в текущем датасете всего 6 точек
     в полосе ±2 °C — модель интерполирует ступеньку по минимуму данных.
  2. LHS не ставит точки ровно на границы домена — края не проверены.
  3. COM-запись T_in / Flow_in не верифицировалась: добавляем обратное чтение.

Три блока точек:
  A) Уплотнение у 100 °C — сетка T_out × несколько (T_in, Flow).
  B) 2³ полный факториал по углам домена.
  C) Центр-грань: середина по двум переменным, крайнее по одной.

Результат: data/phase2_extra_points.xlsx (добавляется к phase1).

HYSYS должен быть запущен с открытым Heater_case.hsc.
"""

import itertools
import sys

import win32com.client as win32

try:
    import openpyxl
    from openpyxl.styles import Font
except ImportError:
    sys.exit("openpyxl не установлен: py -m pip install openpyxl")

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# НАСТРОЙКИ
# ---------------------------------------------------------------------------
STREAM_IN   = "Water In"
STREAM_OUT  = "Water Out"
ENERGY_NAME = "Q-100"

T_IN_RANGE    = (10.0, 40.0)
FLOW_IN_RANGE = (30.0, 80.0)
T_OUT_RANGE   = (60.0, 160.0)

OUT_XLSX = ROOT / "data" / "phase2_extra_points.xlsx"

WRITE_VERIFY_TOL = 0.05   # допуск на обратное чтение (°C / kg/h)


# ---------------------------------------------------------------------------
# Генерация плана
# ---------------------------------------------------------------------------
def build_plan():
    points = []

    # --- A. Уплотнение у 100 °C ---
    t_out_near = [96.0, 98.0, 99.0, 99.5, 100.0, 100.5, 101.0, 102.0, 104.0]
    tin_flow_combos = [(15.0, 40.0), (25.0, 55.0), (35.0, 70.0)]
    for t_out in t_out_near:
        for t_in, flow in tin_flow_combos:
            points.append((t_in, flow, t_out, "near_100"))

    # --- B. Углы домена (полный факториал 2³) ---
    for t_in, flow, t_out in itertools.product(
            T_IN_RANGE, FLOW_IN_RANGE, T_OUT_RANGE):
        points.append((t_in, flow, t_out, "corner"))

    # --- C. Центр-грань ---
    t_in_mid  = sum(T_IN_RANGE) / 2
    flow_mid  = sum(FLOW_IN_RANGE) / 2
    t_out_mid = sum(T_OUT_RANGE) / 2
    for t_in in T_IN_RANGE:
        points.append((t_in, flow_mid, t_out_mid, "face_center"))
    for flow in FLOW_IN_RANGE:
        points.append((t_in_mid, flow, t_out_mid, "face_center"))
    for t_out in T_OUT_RANGE:
        points.append((t_in_mid, flow_mid, t_out, "face_center"))

    return points


# ---------------------------------------------------------------------------
# Прогон через HYSYS
# ---------------------------------------------------------------------------
def run(plan):
    print("Подключаюсь к HYSYS...")
    hysys = win32.GetObject(Class="HYSYS.Application")
    case = hysys.ActiveDocument
    print(f"  Кейс: {case.Title.Value}")

    fs = case.Flowsheet
    s_in  = fs.MaterialStreams.Item(STREAM_IN)
    s_out = fs.MaterialStreams.Item(STREAM_OUT)
    q_stream = fs.EnergyStreams.Item(ENERGY_NAME)

    rows = []
    n = len(plan)
    for i, (t_in, flow, t_out, tag) in enumerate(plan, 1):
        row = {
            "sample": i,
            "tag": tag,
            "T_in_C": round(t_in, 2),
            "Flow_in_kgh": round(flow, 2),
            "T_out_target_C": round(t_out, 2),
            "Q_kJ_per_h": None,
            "VapFrac_out": None,
            "T_out_actual_C": None,
            "T_in_readback_C": None,
            "Flow_readback_kgh": None,
            "write_ok": None,
            "Error": "",
        }
        try:
            s_in.Temperature.SetValue(t_in, "C")
            s_in.MassFlow.SetValue(flow, "kg/h")
            s_out.Temperature.SetValue(t_out, "C")

            t_in_rb  = s_in.Temperature.GetValue("C")
            flow_rb  = s_in.MassFlow.GetValue("kg/h")
            row["T_in_readback_C"]   = round(t_in_rb, 4)
            row["Flow_readback_kgh"] = round(flow_rb, 4)
            row["write_ok"] = (
                abs(t_in_rb - t_in) < WRITE_VERIFY_TOL
                and abs(flow_rb - flow) < WRITE_VERIFY_TOL
            )

            q_val    = q_stream.HeatFlow.GetValue("kJ/h")
            vapfrac  = s_out.VapourFraction.GetValue()
            t_act    = s_out.Temperature.GetValue("C")

            row["Q_kJ_per_h"]     = round(q_val, 1)
            row["VapFrac_out"]    = round(vapfrac, 4)
            row["T_out_actual_C"] = round(t_act, 2)

            status = "OK" if row["write_ok"] else "WRITE MISMATCH"
            print(f"  [{i:3d}/{n}] {tag:12s}  T_in={t_in:5.1f}  Flow={flow:5.1f}  "
                  f"T_out={t_out:6.1f}  ->  Q={q_val:12.1f}  ({status})")

        except Exception as e:
            row["Error"] = str(e)
            print(f"  [{i:3d}/{n}] ОШИБКА: {e}")

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------
def save_xlsx(rows):
    cols = list(rows[0].keys())
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "phase2_extra"
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append([r[c] for c in cols])
    for ci, cn in enumerate(cols, 1):
        letter = ws.cell(1, ci).column_letter
        ws.column_dimensions[letter].width = max(len(cn), 10) + 2
    ws.freeze_panes = "A2"
    OUT_XLSX.parent.mkdir(exist_ok=True)
    wb.save(OUT_XLSX)
    print(f"\nСохранено -> {OUT_XLSX}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plan = build_plan()
    print(f"\nПлан: {len(plan)} точек")
    for tag in ("near_100", "corner", "face_center"):
        print(f"  {tag}: {sum(1 for p in plan if p[3] == tag)}")

    if "--dry-run" in sys.argv:
        print("\n--dry-run: план показан, HYSYS не запускался.")
    else:
        rows = run(plan)
        n_ok  = sum(1 for r in rows if not r["Error"])
        n_err = len(rows) - n_ok
        n_wmm = sum(1 for r in rows if r.get("write_ok") is False)
        print(f"\nИтог: {n_ok} OK, {n_err} ошибок, {n_wmm} write mismatch")
        save_xlsx(rows)
