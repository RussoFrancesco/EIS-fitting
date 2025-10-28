import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit

input_csv = "dataset_sorted.csv"
output_csv = "fit_results.csv"
circuit_str = "L0-R0-p(R1,CPE1)-p(R2,CPE2)-CPE3"

initial_guess = [1e-6, 1e-2, 0.1, 1e-5, 0.9, 1.0, 1e-4, 0.8, 1e-5, 0.5]

df = pd.read_csv(input_csv)
results = []

f_cols = [c for c in df.columns if c.startswith("f_")]
r_cols = [c for c in df.columns if c.startswith("r_")]
i_cols = [c for c in df.columns if c.startswith("i_")]

tmp_circuit = CustomCircuit(circuit_str, initial_guess=initial_guess)
param_names, param_units = tmp_circuit.get_param_names()

print("📋 Parametri del circuito:")
for name, unit in zip(param_names, param_units):
    print(f" - {name} [{unit}]")

for idx, row in df.iterrows():
    cella = row["cell"]
    soh = float(row["soh"])
    soc = float(row["soc"])
    temp = float(row["temperature"])

    freqs = row[f_cols].dropna().to_numpy(dtype=float)
    Z_real = row[r_cols].dropna().to_numpy(dtype=float)
    Z_imag = row[i_cols].dropna().to_numpy(dtype=float)
    Z_imag = -Z_imag
    Z = Z_real + 1j * Z_imag

    if len(freqs) < 3:
        print(f"Curva troppo corta: {cella} | SOH={soh} | SOC={soc}")
        continue

    try:
        circuit = CustomCircuit(circuit_str, initial_guess=initial_guess)
        circuit.fit(freqs, Z)
        Z_fit = circuit.predict(freqs)
        rms_error = np.sqrt(np.mean(np.abs(Z - Z_fit)**2))

        params = circuit.parameters_

        result = {
            "cell": cella,
            "temperature": temp,
            "soh": soh,
            "soc": soc,
            "RMS_error": rms_error,
        }

        for name, val in zip(param_names, params):
            result[name] = val

        results.append(result)
        print(f"Fit completato: {cella} | SOH={soh} | SOC={soc} | err={rms_error:.2e}")

    except Exception as e:
        print(f"Fit fallito per {cella} | SOH={soh} | SOC={soc} — {e}")

fit_df = pd.DataFrame(results)

cols = fit_df.columns.tolist()
unit_row = {col: "" for col in cols}
for name, unit in zip(param_names, param_units):
    if name in unit_row:
        unit_row[name] = unit

fit_df_units = pd.DataFrame([unit_row])
fit_df_final = pd.concat([fit_df_units, fit_df], ignore_index=True)
fit_df_final.to_csv(output_csv, index=False)
