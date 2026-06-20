"""Machine Health Monitor: a Tkinter desktop tool.

Visualises sensor inputs, the model's failure probability, the anomaly flag, the likely failure
type, and an overall machine status (green / amber / red). Includes a 'Simulate stream' mode that
replays real rows from the dataset so you can watch the status change live.

Run:  python -m src.monitor
"""

import random
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd

from .predict import assess, load_artifacts, WATCH, ALARM

SLIDERS = [
    ("air_temp_K", "Air temperature (K)", 295.0, 305.0, 298.0),
    ("process_temp_K", "Process temperature (K)", 305.0, 314.0, 308.0),
    ("rot_speed_rpm", "Rotational speed (rpm)", 1168.0, 2886.0, 1500.0),
    ("torque_Nm", "Torque (Nm)", 3.0, 77.0, 40.0),
    ("tool_wear_min", "Tool wear (min)", 0.0, 253.0, 80.0),
]
COLOURS = {"green": "#1f9d55", "orange": "#e8a020", "red": "#d63b35"}


class Monitor:
    def __init__(self, root):
        self.root = root
        root.title("Machine Health Monitor  -  Predictive Maintenance")
        root.configure(bg="#0f1420")
        try:
            self.art = load_artifacts()
        except Exception as e:
            messagebox.showerror("Models not found", f"Train first: python -m src.train\n\n{e}")
            root.destroy(); return
        try:
            self.data = pd.read_csv("data/ai4i2020.csv")
        except Exception:
            self.data = None
        self.streaming = False
        self.vars = {}
        self._build()
        self.assess_now()

    def _build(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        left = tk.Frame(self.root, bg="#0f1420"); left.grid(row=0, column=0, padx=18, pady=18, sticky="n")
        right = tk.Frame(self.root, bg="#141b2b", bd=0); right.grid(row=0, column=1, padx=18, pady=18, sticky="n")

        tk.Label(left, text="Sensor inputs", fg="#cdd6e6", bg="#0f1420", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        tk.Label(left, text="Product quality type", fg="#9aa6b8", bg="#0f1420").grid(row=1, column=0, sticky="w")
        self.type_var = tk.StringVar(value="L")
        ttk.OptionMenu(left, self.type_var, "L", "L", "M", "H").grid(row=1, column=1, sticky="ew", pady=4)

        for i, (key, label, lo, hi, init) in enumerate(SLIDERS, start=2):
            tk.Label(left, text=label, fg="#9aa6b8", bg="#0f1420").grid(row=i, column=0, sticky="w", pady=(8, 0))
            var = tk.DoubleVar(value=init)
            self.vars[key] = var
            val_lbl = tk.Label(left, text=f"{init:.0f}", fg="#cdd6e6", bg="#0f1420", width=6)
            val_lbl.grid(row=i, column=2, padx=6)
            s = ttk.Scale(left, from_=lo, to=hi, variable=var, orient="horizontal", length=240,
                          command=lambda v, l=val_lbl: l.config(text=f"{float(v):.0f}"))
            s.grid(row=i, column=1, sticky="ew", pady=(8, 0))

        btns = tk.Frame(left, bg="#0f1420"); btns.grid(row=99, column=0, columnspan=3, pady=16, sticky="w")
        tk.Button(btns, text="Assess", command=self.assess_now, bg="#3b6fd4", fg="white", relief="flat", padx=16, pady=6).pack(side="left", padx=4)
        self.stream_btn = tk.Button(btns, text="Simulate stream", command=self.toggle_stream, bg="#26324a", fg="white", relief="flat", padx=16, pady=6)
        self.stream_btn.pack(side="left", padx=4)

        # Status panel
        tk.Label(right, text="Machine status", fg="#cdd6e6", bg="#141b2b", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=18, pady=(16, 8))
        self.light = tk.Canvas(right, width=120, height=120, bg="#141b2b", highlightthickness=0); self.light.pack(pady=6)
        self.circle = self.light.create_oval(15, 15, 105, 105, fill=COLOURS["green"], outline="")
        self.status_lbl = tk.Label(right, text="OK", fg="white", bg="#141b2b", font=("Segoe UI", 22, "bold")); self.status_lbl.pack()
        tk.Label(right, text="Failure probability", fg="#9aa6b8", bg="#141b2b").pack(anchor="w", padx=18, pady=(14, 2))
        self.bar = tk.Canvas(right, width=300, height=22, bg="#0f1420", highlightthickness=0); self.bar.pack(padx=18)
        self.bar_rect = self.bar.create_rectangle(0, 0, 0, 22, fill="#3b6fd4", outline="")
        self.prob_lbl = tk.Label(right, text="0.0%", fg="#cdd6e6", bg="#141b2b"); self.prob_lbl.pack(anchor="e", padx=18)
        self.anom_lbl = tk.Label(right, text="Anomaly: no", fg="#9aa6b8", bg="#141b2b"); self.anom_lbl.pack(anchor="w", padx=18, pady=(10, 0))
        self.type_lbl = tk.Label(right, text="", fg="#9aa6b8", bg="#141b2b"); self.type_lbl.pack(anchor="w", padx=18, pady=(2, 16))

    def reading(self):
        r = {"Type": self.type_var.get()}
        for key, *_ in [(s[0],) for s in SLIDERS]:
            r[key] = float(self.vars[key].get())
        return r

    def assess_now(self):
        res = assess(self.reading(), self.art)
        self.light.itemconfig(self.circle, fill=COLOURS[res["colour"]])
        self.status_lbl.config(text=res["status"])
        p = res["failure_probability"]
        self.bar.coords(self.bar_rect, 0, 0, 300 * p, 22)
        self.bar.itemconfig(self.bar_rect, fill=COLOURS[res["colour"]])
        self.prob_lbl.config(text=f"{p*100:.1f}%")
        self.anom_lbl.config(text=f"Anomaly: {'yes' if res['anomaly'] else 'no'}")
        self.type_lbl.config(text=("" if res["status"] == "OK" else f"If it fails, most likely: {res['likely_failure_type']}"))

    def toggle_stream(self):
        self.streaming = not self.streaming
        self.stream_btn.config(text="Stop stream" if self.streaming else "Simulate stream")
        if self.streaming:
            self._step()

    def _step(self):
        if not self.streaming or self.data is None:
            return
        row = self.data.sample(1).iloc[0]
        self.type_var.set(row["Type"])
        self.vars["air_temp_K"].set(row["Air temperature [K]"])
        self.vars["process_temp_K"].set(row["Process temperature [K]"])
        self.vars["rot_speed_rpm"].set(row["Rotational speed [rpm]"])
        self.vars["torque_Nm"].set(row["Torque [Nm]"])
        self.vars["tool_wear_min"].set(row["Tool wear [min]"])
        self.assess_now()
        self.root.after(1200, self._step)


def main():
    root = tk.Tk()
    Monitor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
