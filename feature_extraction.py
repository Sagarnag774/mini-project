
#======================================================================================
#======================================================================================
#feature_extraction.py
#======================================================================================
#======================================================================================
"""
Feature Extraction Script (Stable v2)
Handles short/flat/NaN EEG segments gracefully.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
from datetime import datetime
import pywt
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================
# 📂 DIRECTORY PATHS
# =========================================================
PREPROCESSED_DIR = r"D:\mini_project\data\pre_processed"
FEATURE_SAVE_DIR = r"D:\mini_project\features"
LOG_DIR = r"D:\mini_project\results\logs"

os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================================================
# ⚙️ FEATURE EXTRACTION FUNCTION
# =========================================================
def extract_features(file_path, fs=256):
    try:
        data = np.load(file_path)
        signal = data["signal"] if "signal" in data else list(data.values())[0]
        signal = np.asarray(signal).flatten()

        # ---- Sanity Checks ----
        if len(signal) < 128:  # too short for FFT/wavelet
            raise ValueError(f"Signal too short ({len(signal)} samples)")
        if np.all(signal == signal[0]):
            raise ValueError("Flat signal (no variation)")

        # Clean NaN and inf values
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        signal = np.clip(signal, -1e6, 1e6)

        features = {}
        fname = os.path.basename(file_path)

        # ============ TIME DOMAIN ============
        features["file_name"] = fname
        features["mean"] = np.mean(signal)
        features["std"] = np.std(signal)
        features["var"] = np.var(signal)
        features["skew"] = skew(signal)
        features["kurtosis"] = kurtosis(signal)

        # ============ FREQUENCY DOMAIN ============
        freqs, psd = welch(signal, fs=fs, nperseg=min(fs*2, len(signal)))
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45)
        }
        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                features[f"{band}_power"] = np.trapz(psd[mask], freqs[mask])
            else:
                features[f"{band}_power"] = 0.0

        # ============ WAVELET FEATURES ============
        try:
            coeffs = pywt.wavedec(signal, 'db4', level=min(4, pywt.dwt_max_level(len(signal), pywt.Wavelet('db4').dec_len)))
            for i, c in enumerate(coeffs):
                features[f"wavelet_energy_L{i}"] = np.sum(np.square(c))
                features[f"wavelet_mean_L{i}"] = np.mean(np.abs(c))
        except Exception:
            features["wavelet_energy_L0"] = np.sum(signal**2)
            features["wavelet_mean_L0"] = np.mean(np.abs(signal))

        # ============ ENTROPY ============
        hist, _ = np.histogram(signal, bins=100, density=True)
        hist += 1e-10  # prevent log(0)
        features["shannon_entropy"] = entropy(hist)
        features["signal_energy"] = np.sum(signal ** 2)
        features["zero_cross_rate"] = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)

        return features, None

    except Exception as e:
        return None, f"{file_path}: {str(e)}"

# =========================================================
# 🧠 MAIN EXECUTION
# =========================================================
def main():
    files = [os.path.join(PREPROCESSED_DIR, f) for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz")]
    all_features = []
    logs = []

    print(f"🔍 Found {len(files)} EEG files.")
    print("⚙️ Starting feature extraction...")

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_features, file): file for file in files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                features, error = future.result()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if features:
                    all_features.append(features)
                    logs.append({"File": os.path.basename(file), "Status": "Success", "Timestamp": timestamp})
                    print(f"✅ Processed: {os.path.basename(file)}")
                else:
                    logs.append({"File": os.path.basename(file), "Status": f"Failed - {error}", "Timestamp": timestamp})
                    print(f"❌ Failed: {os.path.basename(file)} | {error}")

            except Exception as e:
                logs.append({"File": os.path.basename(file), "Status": f"Exception - {str(e)}", "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                print(f"⚠️ Exception: {os.path.basename(file)} | {e}")

    # Save outputs
    if all_features:
        df = pd.DataFrame(all_features)
        feature_path = os.path.join(FEATURE_SAVE_DIR, "features.csv")
        df.to_csv(feature_path, index=False)
        print(f"\n📁 Features saved: {feature_path}")

    log_path = os.path.join(LOG_DIR, "extraction_log.xlsx")
    pd.DataFrame(logs).to_excel(log_path, index=False)
    print(f"🧾 Log saved: {log_path}")

    print("\n✅ Feature extraction completed.")

# =========================================================
# 🚀 RUN
# =========================================================
if __name__ == "__main__":
    main()
