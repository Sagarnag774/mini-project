#=========================================================
#=========================================================
#generate_annotations.py
#=========================================================
#=========================================================
import os
import re
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"D:\mini_project\data\raw\i-care_dataset\training"   # <<< CHANGE THIS TO YOUR RAW DATA FOLDER
ANNOTATIONS_CSV = r"D:\mini_project\results\logs\annotations.csv"
PATIENT_SUMMARY_CSV = r"D:\mini_project\results\logs\patient_summary.csv"

# Create output dirs
os.makedirs("plots", exist_ok=True)

# ============================================================
# Helper Functions
# ============================================================

def parse_header_file(hea_path):
    """
    Extracts key metadata from .hea file.
    Returns dictionary of:
      - patient_id
      - sampling_rate
      - num_channels
      - signal_names
      - duration_sec
    """
    meta = {
        "patient_id": None,
        "sampling_rate": None,
        "num_channels": None,
        "signal_names": [],
        "duration_sec": None
    }

    try:
        with open(hea_path, 'r') as f:
            lines = f.readlines()

        # Example first line format:
        # record 14 500 60000
        first = lines[0].strip().split()
        if len(first) >= 4:
            meta["record_name"] = first[0]
            meta["num_channels"] = int(first[1])
            meta["sampling_rate"] = float(first[2])
            total_samples = int(first[3])
            meta["duration_sec"] = total_samples / meta["sampling_rate"]

        # Try to extract signal names (channel labels)
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 1:
                meta["signal_names"].append(parts[0])

        # Infer patient ID (e.g., patient_01, p01_, etc.)
        # Custom logic for ICARE-like datasets:
        fname = os.path.basename(hea_path)
        guessed = re.findall(r"\d+", fname)
        meta["patient_id"] = guessed[0] if guessed else "unknown"

        return meta

    except Exception as e:
        print(f"Error reading {hea_path}: {e}")
        return None


def extract_mat_info(mat_path):
    """
    Checks if .mat file is loadable & extracts variable keys.
    Does NOT load full data to avoid memory issues.
    """
    try:
        mat = sio.whosmat(mat_path)
        var_names = [v[0] for v in mat]
        return var_names
    except Exception as e:
        return ["LOAD_ERROR"]


# ============================================================
# Main Extraction Loop
# ============================================================

annotations = []

for root, _, files in os.walk(DATA_DIR):
    mat_files = [f for f in files if f.endswith(".mat")]

    for mat_file in mat_files:
        mat_path = os.path.join(root, mat_file)

        # Find corresponding .hea
        hea_file = mat_file.replace(".mat", ".hea")
        hea_path = os.path.join(root, hea_file)

        # Check existence
        if not os.path.exists(hea_path):
            annotations.append({
                "filename_mat": mat_file,
                "filename_hea": None,
                "status": "missing_header",
                "patient_id": None,
                "num_channels": None,
                "sampling_rate": None,
                "duration_sec": None,
                "signal_names": None,
                "mat_variables": None
            })
            continue

        # Parse header metadata
        meta = parse_header_file(hea_path)
        mat_vars = extract_mat_info(mat_path)

        annotations.append({
            "filename_mat": mat_file,
            "filename_hea": hea_file,
            "status": "ok" if meta is not None else "corrupted",
            "patient_id": meta["patient_id"],
            "num_channels": meta["num_channels"],
            "sampling_rate": meta["sampling_rate"],
            "duration_sec": meta["duration_sec"],
            "signal_names": ";".join(meta["signal_names"]),
            "mat_variables": ";".join(mat_vars)
        })

# ============================================================
# Save annotations.csv
# ============================================================

df = pd.DataFrame(annotations)
df.to_csv(ANNOTATIONS_CSV, index=False)
print(f"Saved: {ANNOTATIONS_CSV}")
df.head()
# ============================================================
# PATIENT-LEVEL SUMMARY
# ============================================================

# Filter valid entries
valid = df[df["status"] == "ok"].copy()

# Guarantee numerical
valid["duration_sec"] = pd.to_numeric(valid["duration_sec"], errors="coerce")

patient_summary = valid.groupby("patient_id").agg({
    "filename_mat": "count",
    "num_channels": "max",
    "sampling_rate": "median",
    "duration_sec": "sum"
}).reset_index()

patient_summary.rename(columns={
    "filename_mat": "num_files",
    "duration_sec": "total_duration_sec"
}, inplace=True)

patient_summary["total_hours"] = patient_summary["total_duration_sec"] / 3600.0

# Save summary CSV
patient_summary.to_csv(PATIENT_SUMMARY_CSV, index=False)
print(f"Saved: {PATIENT_SUMMARY_CSV}")

patient_summary
# ============================================================
# PLOTS
# ============================================================

# Plot: number of files per patient
plt.figure(figsize=(10,6))
plt.bar(patient_summary["patient_id"], patient_summary["num_files"])
plt.xlabel("Patient ID")
plt.ylabel("Number of EEG Files")
plt.title("Files per Patient")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/files_per_patient.png")
plt.show()

# Plot: total hours per patient
plt.figure(figsize=(10,6))
plt.bar(patient_summary["patient_id"], patient_summary["total_hours"])
plt.xlabel("Patient ID")
plt.ylabel("Total Duration (hours)")
plt.title("EEG Recording Hours per Patient")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/hours_per_patient.png")
plt.show()
