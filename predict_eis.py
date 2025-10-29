import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist


input_csv = "dataset_sorted.csv"
circuit_str = "L0-R0-p(R1,CPE1)-p(R2,CPE2)-CPE3"
initial_guess = [1e-6, 1e-2, 0.1, 1e-5, 0.9, 1.0, 1e-4, 0.8, 1e-5, 0.5]

df = pd.read_csv(input_csv)

row = df.iloc[0]
cella = row["cell"]
soh = float(row["soh"])
soc = float(row["soc"])
temp = float(row["temperature"])
idx = int(row["id"])

f_cols = [c for c in df.columns if c.startswith("f_")]
r_cols = [c for c in df.columns if c.startswith("r_")]
i_cols = [c for c in df.columns if c.startswith("i_")]

freqs = row[f_cols].dropna().to_numpy(dtype=float)
Z_real = row[r_cols].dropna().to_numpy(dtype=float)
Z_imag = row[i_cols].dropna().to_numpy(dtype=float)
Z_imag = -Z_imag
Z = Z_real + 1j * Z_imag

circuit = CustomCircuit(circuit_str, initial_guess=initial_guess)
circuit.fit(freqs, Z)

initial_prediction = circuit.predict(freqs)

Z2 = (0.005 + Z.real)*1.5 + 1.5j*Z.imag
circuit.fit(freqs, Z2)

prediction = circuit.predict(freqs)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(initial_prediction.real, -initial_prediction.imag, 'o', label='Measured Data')
ax.plot(prediction.real, -prediction.imag, '--', label='Fitted Circuit')
ax.set_xlabel('Z\' (Ohm)')
ax.set_ylabel('-Z\'\' (Ohm)')
ax.set_title('Nyquist Plot')
ax.legend()
ax.grid(True)
plt.axis('equal')
plt.show()

