#======================================================================================
#======================================================================================
#preprocessing.py
#======================================================================================
#======================================================================================

# -------------------------------------------------------------
# EEG Preprocessing Script for I-CARE Dataset
# Author: Tigerram
# Version: 3.0 (Storage Optimized)
# -------------------------------------------------------------

import os
import mne
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.io import loadmat
from scipy.signal import welch

# ---------------- USER CONFIGURATION ---------------- #
RAW_DATA_DIR = r"D:\mini_project\data\raw\i-care_dataset"
PROCESSED_DIR = r"D:\mini_project\data\pre_processed"
LOG_DIR = r"D:\mini_project\results\logs"

# Choose processing mode:
#   "A" = downsampled + float16 waveform
#   "B" = feature extraction only (tiny output)
MODE = "A"

# Target sampling frequency for downsampling (Hz)
TARGET_FS = 25.0

# ---------------- DIRECTORY SETUP ---------------- #
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
log_entries = []

# ---------------- UTILITIES ---------------- #
def is_eeg_file(file_name):
    return file_name.endswith("EEG.hea") or file_name.endswith("EEG.mat")


def preprocess_eeg(file_path):
    """Load and filter EEG data."""
    try:
        if file_path.endswith(".mat"):
            mat = loadmat(file_path)
            key = next((k for k in mat.keys() if not k.startswith("__")), None)
            eeg_data = mat[key]
            if isinstance(eeg_data, dict):
                eeg_data = eeg_data.get("data") or eeg_data.get("val")
            data = np.array(eeg_data, dtype=np.float32)
            data = np.nan_to_num(data)

            fs = 250.0  # I-CARE default
            info = mne.create_info(ch_names=[f"ch{i}" for i in range(data.shape[0])],
                                   sfreq=fs, ch_types="eeg")
            raw = mne.io.RawArray(data, info, verbose=False)
            raw.filter(0.5, 45., fir_design='firwin', verbose=False)
            raw.notch_filter(freqs=[50], verbose=False)
            raw.set_eeg_reference('average', verbose=False)
            data, _ = raw.get_data(return_times=True)
            return data.astype(np.float32), fs

        elif file_path.endswith(".edf"):
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            fs = raw.info["sfreq"]
            raw.filter(0.5, 45., fir_design='firwin', verbose=False)
            raw.notch_filter(freqs=[50], verbose=False)
            raw.set_eeg_reference('average', verbose=False)
            data, _ = raw.get_data(return_times=True)
            return data.astype(np.float32), fs

        else:
            return None, None
    except Exception as e:
        print(f"[ERROR] Could not process {file_path}: {e}")
        return None, None


def reduce_and_save(data, fs, save_path, mode="A"):
    """Storage-efficient save: Mode A = waveform, Mode B = features."""
    try:
        if mode == "A":
            # --- Downsample + Quantize ---
            decim = int(fs / TARGET_FS)
            data = data[:, ::decim]
            fs = TARGET_FS
            data = data.astype("float16")
            np.savez_compressed(save_path.replace(".npy", "_reduced.npz"),
                                data=data, fs=fs)
            return True

        elif mode == "B":
            # --- Feature Extraction (Welch PSD Bands) ---
            psd, freqs = welch(data, fs=fs, nperseg=int(fs*2))
            bands = {'delta': (0.5, 4), 'theta': (4, 8),
                     'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
            feats = {}
            for name, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs <= hi)
                feats[name + '_mean'] = psd[:, mask].mean(axis=1).astype('float32')
                feats[name + '_std'] = psd[:, mask].std(axis=1).astype('float32')

            np.savez_compressed(save_path.replace(".npy", "_features.npz"),
                                **feats, fs=fs)
            return True

    except Exception as e:
        print(f"[SAVE ERROR] {os.path.basename(save_path)}: {e}")
        return False


# ---------------- MAIN PIPELINE ---------------- #
print("🔍 Scanning dataset...")

for root, _, files in os.walk(RAW_DATA_DIR):
    for file in files:
        if not is_eeg_file(file):
            continue

        file_path = os.path.join(root, file)
        rel_path = os.path.relpath(file_path, RAW_DATA_DIR)

        base_name = os.path.splitext(file)[0].replace('.', '_')
        save_name = base_name + ".npy"
        save_path = os.path.join(PROCESSED_DIR, save_name)

        if os.path.exists(save_path.replace(".npy", "_reduced.npz")) or \
           os.path.exists(save_path.replace(".npy", "_features.npz")):
            print(f"Skipping already processed: {rel_path}")
            continue

        print(f"Processing: {rel_path}")
        data, fs = preprocess_eeg(file_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if data is not None:
            success = reduce_and_save(data, fs, save_path, MODE)
            status = "Processed" if success else "SaveFailed"
            log_entries.append({
                "timestamp": timestamp,
                "file_name": file,
                "status": status,
                "channels": data.shape[0],
                "samples": data.shape[1],
                "sampling_rate": fs,
                "save_path": save_path
            })
        else:
            log_entries.append({
                "timestamp": timestamp,
                "file_name": file,
                "status": "Failed",
                "channels": None,
                "samples": None,
                "sampling_rate": None,
                "save_path": None
            })

# ---------------- LOGGING ---------------- #
log_df = pd.DataFrame(log_entries)
timestamp_now = datetime.now().strftime('%Y%m%d_%H%M%S')
main_log = os.path.join(LOG_DIR, f"preprocessing_log_{timestamp_now}.xlsx")
error_log = os.path.join(LOG_DIR, f"preprocessing_error_log_{timestamp_now}.xlsx")

log_df.to_excel(main_log, index=False, engine='openpyxl')
errors = log_df[log_df["status"].isin(["Failed", "SaveFailed"])]
if not errors.empty:
    errors.to_excel(error_log, index=False, engine='openpyxl')
    print(f"⚠️ Error log created: {error_log}")

processed = len(log_df[log_df['status'].str.contains('Processed')])
failed = len(log_df[log_df['status'].str.contains('Failed')])

print("\nPreprocessing complete.")
print(f"Output saved in: {PROCESSED_DIR}")
print(f"Log file: {main_log}")
print(f"Summary: {processed} processed, {failed} failed.")
print(f"Mode: {MODE} | Target FS: {TARGET_FS} Hz")








