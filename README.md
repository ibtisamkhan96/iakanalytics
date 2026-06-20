# Predictive Maintenance for Manufacturing Machines

An end-to-end machine-learning system that predicts machine failure from sensor data, flags anomalies, classifies the likely failure mode, and presents it as a live **Machine Health Monitor** desktop tool. Built on the real UCI AI4I 2020 dataset (10,000 machine cycles, a realistic 3.4% failure rate).

It covers the full loop an AI and data-science engineer owns in a manufacturing setting: a clean **data pipeline**, **model training, evaluation and benchmarking**, a saved deployable model, and a **simple UI** for data, model outputs and machine status.

## Results (verified on a held-out 25% test set)

| Model | CV PR-AUC | CV ROC-AUC |
|------|-----------|------------|
| Logistic Regression | 0.494 | 0.925 |
| Random Forest | 0.859 | 0.973 |
| **Histogram Gradient Boosting (chosen)** | **0.864** | 0.971 |

- **Test ROC-AUC 0.981, test PR-AUC 0.896** on the chosen model (PR-AUC is the honest metric here because only 3.4% of cycles fail).
- **Anomaly detector** (unsupervised Isolation Forest) flags about 3.5% of cycles and catches a share of failures with no labels at all, as an independent novelty signal.
- **Failure-type classifier**: 90.3% cross-validated accuracy across the heat, power, overstrain and tool-wear failure modes.

Plots (ROC, precision-recall) and full metrics are in `reports/`.

## System design

```
 Sensors (air/process temp, speed, torque, tool wear, quality type)
        │
        ▼
 pipeline.py   load -> data-quality checks -> feature engineering (power, temp gradient) -> (X, y)
        │
        ▼
 train.py      failure classifier  +  anomaly detector  +  failure-type classifier
        │        (benchmarked, cross-validated, saved to models/)
        ▼
 predict.py    one reading -> failure probability, anomaly flag, likely type -> status (OK / WATCH / ALARM)
        │
        ▼
 monitor.py    Tkinter desktop tool   (or any service / dashboard via predict.assess)
```

`predict.assess(reading)` is the clean integration point: a service, PLC bridge, or dashboard calls it the same way the desktop tool does.

## The data pipeline

- **Quality checks:** missing values, duplicates, and physical range checks on every sensor.
- **Feature engineering:** mechanical power (torque times angular velocity) and the process-to-air temperature gradient, the physics behind the real power and heat-dissipation failure modes.
- **Leakage control:** the five failure-mode flags are components of the label, so they are excluded from the model inputs and used only as the failure-type target.

## Run it

```bash
pip install -r requirements.txt
python -m src.train       # trains, benchmarks, evaluates, saves models + reports
python -m src.predict     # assess two example readings from the command line
python -m src.monitor     # launch the Machine Health Monitor desktop tool (Tkinter)
```

The desktop tool has sliders for each sensor, an Assess button, and a Simulate stream mode that replays real cycles so you can watch the status light change. Trained models are committed, so the tool runs without retraining.

## Honest notes

- AI4I 2020 is a realistic synthetic benchmark, not one factory's logs, so absolute numbers would differ on real plant data. The pipeline, the metrics methodology, and the decision logic are what transfer.
- The unsupervised anomaly detector is a weak independent signal on this dataset, kept deliberately as a second opinion alongside the supervised model, not as the primary alarm.
- PR-AUC and recall on the failure class matter more than accuracy here, because a model that always predicts "no failure" would be 96.6% accurate and useless.

## Stack

Python, NumPy, Pandas, scikit-learn (Histogram Gradient Boosting, Random Forest, Logistic Regression, Isolation Forest), Matplotlib, joblib, Tkinter. Data: UCI AI4I 2020 Predictive Maintenance Dataset.

---

*Built by Ibtisam Ahmed Khan, materials engineer and data and AI practitioner. [linkedin.com/in/ibtisam-ahmed-khan](https://linkedin.com/in/ibtisam-ahmed-khan)*
