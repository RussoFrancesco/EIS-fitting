import pandas as pd
import glob
import re
import os

folder_path = "dataset"  
output_csv = "complete_dataset.csv"


files = glob.glob(os.path.join(folder_path, "*.xlsx"))

all_data = []
max_points = 0

for idx, file in enumerate(sorted(files), start=1):
    filename = os.path.basename(file).replace(".xlsx", "")

    match = re.match(r"(Cell\d+)_(\d+)SOH_(\d+)degC_(\d+)SOC", filename)
    if not match:
        print(f"⚠️ Nome file non conforme: {filename}")
        continue

    cella, soh, temp, soc = match.groups()

    df = pd.read_excel(file, header=None, names=["f", "r", "i"])
    n_points = len(df)
    max_points = max(max_points, n_points)

    row = {"id": idx, "cell": cella}

    for j, (_, line) in enumerate(df.iterrows(), start=1):
        row[f"f_{j}"] = line["f"]
        row[f"r_{j}"] = line["r"]
        row[f"i_{j}"] = line["i"]

    row["temperature"] = temp
    row["soh"] = soh
    row["soc"] = soc

    all_data.append(row)

merged_df = pd.DataFrame(all_data)

ordered_cols = ["id", "cell"]
for j in range(1, max_points + 1):
    ordered_cols += [f"f_{j}", f"r_{j}", f"i_{j}"]
ordered_cols += ["temperature", "soh", "soc"]

merged_df = merged_df.reindex(columns=ordered_cols)

merged_df.to_csv(output_csv, index=False)

