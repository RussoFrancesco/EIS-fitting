import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from impedance.validation import linKK
from impedance.models.circuits import CustomCircuit

# === Parametri ===
input_csv = "dataset_sorted.csv"
circuit_str = "L0-R0-p(R1,CPE1)-p(R2,CPE2)-CPE3"
initial_guess = [1e-6, 1e-2, 0.1, 1e-5, 0.9, 1.0, 1e-4, 0.8, 1e-5, 0.5]

# === Caricamento dati ===
df = pd.read_csv(input_csv)
f_cols = [c for c in df.columns if c.startswith("f_")]
r_cols = [c for c in df.columns if c.startswith("r_")]
i_cols = [c for c in df.columns if c.startswith("i_")]

circuit = CustomCircuit(circuit_str, initial_guess=initial_guess)

results = []

# === Loop su tutte le curve ===
for _, row in df.iterrows():
    cella = row["cell"]
    soh = float(row["soh"])
    soc = float(row["soc"])
    temp = float(row["temperature"])
    idx = int(row["id"])

    # --- Costruisci Z ---
    freqs = row[f_cols].dropna().to_numpy(dtype=float)
    Z_real = row[r_cols].dropna().to_numpy(dtype=float)
    Z_imag = -row[i_cols].dropna().to_numpy(dtype=float)
    n = min(len(freqs), len(Z_real), len(Z_imag))
    freqs, Z_real, Z_imag = freqs[:n], Z_real[:n], Z_imag[:n]
    Z = Z_real + 1j * Z_imag

    # --- Fit del circuito ---
    try:
        circuit.fit(freqs, Z)
        Z_fit = circuit.predict(freqs)
        rms_error = np.sqrt(np.mean(np.abs(Z - Z_fit)**2))

        # --- Validazione KK ---
        M, mu, Z_linKK, res_real, res_imag = linKK(freqs, Z, c=0.5, max_M=100,
                                                   fit_type='complex', add_cap=True)
        
        results.append({
            "id": idx,
            "cell": cella,
            "soh": soh,
            "soc": soc,
            "temperature": temp,
            "M": M,
            "mu": mu,
            "RMS_error": rms_error
        })
        print(f"✓ Fitted curve {idx} | mu={mu:.3f} | RMS={rms_error:.4f}")

    except Exception as e:
        print(f"⚠️ Errore nel fit della curva {idx}: {e}")

# === Conversione risultati ===
results_df = pd.DataFrame(results)

# === Filtraggio delle curve buone / cattive ===
good_fits = results_df[(results_df["mu"] < 0.85) & (results_df["RMS_error"] <= 0.01)]
bad_fits = results_df[(results_df["mu"] >= 0.9) | (results_df["RMS_error"] > 0.01)]

# === Salvataggio su CSV ===
good_fits.to_csv("good_fits.csv", index=False)
bad_fits.to_csv("bad_fits.csv", index=False)

print(f"\n✅ Salvate {len(good_fits)} curve buone in 'good_fits.csv'")
print(f"❌ Salvate {len(bad_fits)} curve cattive in 'bad_fits.csv'")

# === (Opzionale) Visualizzazione distribuzione qualità ===
plt.figure(figsize=(6,5))
plt.scatter(results_df["mu"], results_df["RMS_error"], c='tab:blue', label='All curves')
plt.axvline(0.85, color='g', linestyle='--', label='μ=0.85')
plt.axvline(0.9, color='r', linestyle='--', label='μ=0.9')
plt.xlabel('μ (Lin-KK)')
plt.ylabel('RMS Error')
plt.title('Distribuzione qualità dei fit')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
