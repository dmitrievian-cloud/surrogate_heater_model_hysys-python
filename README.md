# Aspen HYSYS Heater Surrogate Model

A machine learning surrogate model for a simple Aspen HYSYS heater
(Water In → E-100 → Water Out). Automates data generation via Python/COM,
trains and validates a model that replaces repeated HYSYS runs for fast
what-if analysis — proof of concept for scaling the same approach to more
complex units (e.g. an FCC).

## Why

Process simulators like HYSYS are accurate but slow to query repeatedly.
A surrogate model learns the simulator's input-output behavior once, then
answers new queries in milliseconds instead of seconds — enabling
optimization, Monte Carlo analysis, and real-time operator advisory tools
that would be impractical running the full simulator thousands of times.

## HYSYS model

The source simulation case is in `hysys_case/heater_case.hsc` (Aspen HYSYS).
A simple Heater block (E-100) with a water stream (Water In → Water Out)
and its energy stream (Q-100) — the model `heater_01_generate_dataset.py`
drives via COM automation.

## Pipeline

1. **`src/heater_01_generate_dataset.py`** — connects to a running HYSYS
   case via COM automation (`pywin32`), samples the input space (T_in,
   Flow_in, T_out) using Latin Hypercube Sampling (DoE), and records the
   simulator's response. Output: `data/phase1_heater_dataset.xlsx`.
2. **`src/heater_02_train_model.py`** — trains and compares Linear
   Regression, Gradient Boosting, a plain Gaussian Process and the
   Enthalpy GP (below), evaluated with 5-fold cross-validation
   (out-of-fold predictions, not a single train/test split). An earlier
   `Dist_to_boiling_C = T_out - 100` feature was removed: it is `T_out`
   shifted by a constant, so it carries no new information (trees ignore the
   shift, and for linear regression it is a duplicate column).
3. **`src/heater_03_deploy_model.py`** — trains the final model (Enthalpy
   GP) on the full dataset and saves it to `models/surrogate_heater_model.pkl`.
   The model class and `predict_Q(T_in, Flow_in, T_out, return_std=False)`
   live in `src/surrogate_model.py`, shared by scripts 02–05.
4. **`src/heater_04_use_surrogate_batch.py`** — batch what-if / sensitivity
   analysis using the saved model (e.g. Q vs T_out across several T_in
   values), no HYSYS required.
5. **`src/heater_05_use_surrogate_interactive.py`** — interactive console
   querying of the saved model (enter T_in, Flow_in, T_out, get Q ± σ back
   instantly).
6. **`src/heater_06_optimize.py`** — optimization on top of the surrogate
   (`scipy.optimize.brentq`) under a conservative constraint
   `Q + 2σ <= Q_max`: maximum flow for a given T_out and heater limit, and
   maximum reachable T_out for a given flow (it reports when the limit falls
   short of the latent heat at 100 °C).

`predict_Q` warns (`RuntimeWarning`) when an input is outside the training
ranges, which are stored in the model file. The sensitivity plot shows a
±2σ band.

## Results

| Model | R² (5-fold CV) | MAE (kJ/h) |
|---|---|---|
| Linear Regression | 0.71 | ~30,100 |
| Gradient Boosting  | 0.95 | ~3,600 |
| Gaussian Process (Matern 5/2, ARD) | 0.99 | ~2,700 |
| **Enthalpy GP** (deployed) | 1.00 | ~5 |

**Why a plain GP is not enough.** It is more accurate than Gradient
Boosting and its predictive std flags extrapolation, but it does **not**
flag the boiling-point step: in CV its std is nearly flat across T_out
(~5,000 kJ/h) while the real error at T_out 98–102 °C is ~22,000 kJ/h
(correlation of std and error: 0.07). A stationary kernel cannot represent
a step.

**Enthalpy GP.** Since Q = m·(h_out − h_in), the specific heat duty
q = Q / Flow depends only on the temperatures (checked on the data: the
flow coefficient is ≈ 0, and dq/dT_out ≈ 4.2 kJ/kg/K in the liquid, i.e.
cp of water). The GP is therefore trained on q with features
`[T_in, T_out, is_vapor]`, where `is_vapor = T_out >= 100 °C`; the latent
heat jump lives entirely in that flag, and the smooth part is left to the
GP. Result: R² = 1.0000, MAE ≈ 5 kJ/h in 5-fold CV, with similar error in
every T_out band including 98–102 °C. The predictive std (~200 kJ/h) is
conservative (real errors are ~5 kJ/h) but uniform and no longer misleading;
it still grows when extrapolating (e.g. ~1,600 kJ/h at T_out = 300 °C).
This is not the removed `Dist_to_boiling_C`: a step flag carries information
a constant shift does not.

![Parity plot](figures/phase2_parity_plot.png)
![Sensitivity analysis](figures/sensitivity_Q_vs_Tout.png)

## Known limitations

- **Phase-change point.** Exactly at the boiling point, a temperature-and-
  pressure specification does not determine vapor fraction (the stream sits
  on the saturation line), so the true value is ambiguous. The Enthalpy GP
  handles it with a hard threshold `T_out >= T_sat` (100 °C); the nearest
  training points are at 99.9 and 100.4 °C, so within a few tenths of a
  degree of the threshold the result is only as good as that convention.
- **Fixed pressure.** `T_sat = 100 °C` is a constant (the case runs at
  ~101 kPa). A different pressure needs a new T_sat and regenerated data.
- **Limited extrapolation.** The model was trained on T_in ≈ 10–40 °C,
  T_out ≈ 60–160 °C, flow ≈ 30–80 kg/h (150 LHS points). Outside that domain
  the GP std grows, but the mean reverts toward the prior and loses accuracy;
  check `return_std` or the ranges in `heater_01_generate_dataset.py`.
- **Heater-specific trick.** Dividing by flow and the phase flag use known
  physics of this unit; the same recipe for an FCC would need its own
  physics-informed target and features.

## How to run

```bash
pip install -r requirements.txt

# with HYSYS open and the heater model loaded:
python src/heater_01_generate_dataset.py
python src/heater_02_train_model.py
python src/heater_03_deploy_model.py

# afterwards, HYSYS is no longer required:
python src/heater_04_use_surrogate_batch.py
python src/heater_05_use_surrogate_interactive.py
python src/heater_06_optimize.py
```

## Roadmap

- Scale the same pipeline to a more complex unit (FCC).
- Multi-variable optimization (e.g. cost of Q vs throughput) with
  `scipy.optimize.minimize`; the current script solves 1-D problems only.