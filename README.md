# ML Surrogate Models for Aspen HYSYS Simulations

A pipeline that builds machine learning surrogate models of Aspen HYSYS process
simulations: a trained model replaces repeated HYSYS runs for fast "what-if"
analysis, optimization, and sensitivity studies. Validated on a simple water
heater, designed to scale toward more complex units (e.g. FCC).

## Why

Process simulators like HYSYS are accurate but slow to query repeatedly.
Surrogate models learn the simulator's input-output behavior once, then
answer new queries in milliseconds instead of seconds — enabling optimization,
Monte Carlo analysis, and real-time operator advisory tools that would be
impractical running the full simulator thousands of times.

## Pipeline

1. **Data generation** (`src/01_generate_dataset.py`) — connects to a running
   HYSYS case via COM automation (`pywin32`), samples the input space using
   Latin Hypercube Sampling (DoE), and records the simulator's response.
2. **Model training** (`src/02_train_model.py`) — trains and compares Linear
   Regression and Gradient Boosting, evaluated with 5-fold cross-validation
   (out-of-fold predictions, not a single train/test split).
3. **Deployment** (`src/03_deploy_model.py`) — trains the final model on the
   full dataset and saves it for reuse.
4. **Usage** (`src/04_use_surrogate.py`) — interactive and batch querying of
   the saved model, without needing HYSYS open.

## Results

| Model | R² (5-fold CV) | MAE (kJ/h) |
|---|---|---|
| Linear Regression | 0.71 | ~30,100 |
| Gradient Boosting  | 0.95 | ~3,600 |

![Parity plot](figures/phase2_parity_plot.png)
![Sensitivity analysis](figures/sensitivity_Q_vs_Tout.png)

## Known limitations

- **Phase-change discontinuity.** Near the water's boiling point, a
  temperature-and-pressure specification no longer uniquely determines vapor
  fraction (the stream sits on the saturation line). The underlying
  relationship has a genuine step discontinuity there, which no regression
  model — including this one — can represent exactly at that single point.
  An explicit `Dist_to_boiling_C` feature was added to help the model
  separate the two regimes, but predictions queried exactly at the boiling
  point should not be trusted.
- **No extrapolation beyond the training domain.** The model was trained
  on inputs within specific ranges (e.g. mass flow 30–80 kg/h). Tree-based
  models like Gradient Boosting cannot extrapolate past the ranges they were
  trained on — queries outside that domain silently degrade in accuracy.
  Always check inputs against the ranges used in `01_generate_dataset.py`.

## How to run

```bash
pip install -r requirements.txt

# with HYSYS open and the model loaded:
python src/01_generate_dataset.py
python src/02_train_model.py
python src/03_deploy_model.py

# afterwards, HYSYS is no longer required:
python src/04_use_surrogate.py
```

## Roadmap

- Scale the same pipeline to a more complex unit (FCC).
- Replace Gradient Boosting with Gaussian Process Regression to get
  uncertainty estimates alongside predictions (flagging low-confidence
  regions, such as near the phase-change boundary, automatically).
- Wrap the surrogate in an optimization loop (e.g. `scipy.optimize`) for
  real "what-if" decision support.