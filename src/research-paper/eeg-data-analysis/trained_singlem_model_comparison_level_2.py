import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy import signal

# Point Python to the SingLEM submodule
sys.path.append(os.path.join(os.path.dirname(__file__), "SingLEM"))
from SingLEM.model import Config, EEGEncoder

# --- CONFIGURATION ---
DATA_DIR = "data/level_2"
MODEL_WEIGHTS_PATH = "singlem_binary_head.pth"
TARGET_TEST = "Stroop"
CONDITIONS = ["Silent", "WhiteNoise", "Music"]
EEG_COLUMN = "eegRawValueVolts"

# Strict regex for the current analysis session
TARGET_PARTICIPANT_REGEX = r"^Andy$"

# DSP Constants
ORIGINAL_FS = 512
TARGET_FS = 128
WINDOW_SECONDS = 5
TOKENS_PER_WINDOW = WINDOW_SECONDS
SAMPLES_PER_TOKEN = TARGET_FS
DISCARD_SECONDS = 60  # Must match training to ensure we analyze the same brain state

# Updated Regex to capture the participant name group
FILENAME_REGEX = re.compile(
    r"^\d{8}_\d{4}_(?P<participant>[A-Za-z0-9]+)_EEG_(?P<test>Stroop|Typing)_(?P<condition>Silent|WhiteNoise|Music)\.csv$",
    re.IGNORECASE,
)


def preprocess_eeg(raw_volts):
    """
    Standard SingLEM preprocessing:
    Isolates 0.5-50Hz, removes 60Hz hum, decimates by 4x, and Z-normalizes.
    """
    b, a = signal.butter(4, [0.5, 50.0], btype="bandpass", fs=ORIGINAL_FS)
    filtered = signal.filtfilt(b, a, raw_volts)

    b_notch, a_notch = signal.iirnotch(60.0, 30.0, fs=ORIGINAL_FS)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)

    resampled = signal.decimate(filtered, q=4)
    scaled = resampled * 1e4
    normalized = (scaled - np.mean(scaled)) / (np.std(scaled) + 1e-8)
    return normalized


# --- MODEL ARCHITECTURE ---
class BinaryAnxietyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BinaryAnxietyClassifier, self).__init__()
        config = Config()
        config.mask_prob = 0.0
        self.feature_extractor = EEGEncoder(config)

        self.classifier = nn.Sequential(
            nn.Linear(5 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features, _, _ = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


def load_model():
    """Initializes model and attempts to load trained weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryAnxietyClassifier(num_classes=2)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print(f"✅ Loaded weights: {MODEL_WEIGHTS_PATH}")
    else:
        print(f"⚠️ Warning: {MODEL_WEIGHTS_PATH} not found. Using random weights.")

    model.to(device)
    model.eval()  # CRITICAL: Disables Dropout/BatchNorm for consistent inference
    return model, device


def analyze_eeg_files(model, device):
    """Parses files, applies participant filtering, and generates load predictions."""
    raw_data = {cond: [] for cond in CONDITIONS}
    participant_filter = re.compile(TARGET_PARTICIPANT_REGEX)

    for filename in os.listdir(DATA_DIR):
        match = FILENAME_REGEX.match(filename)
        if not match:
            continue

        # Extract info from filename using capture groups
        file_participant = match.group("participant")
        file_test = match.group("test").capitalize()
        condition = match.group("condition")

        # FILTER 1: Participant Match
        if not participant_filter.search(file_participant):
            continue

        # FILTER 2: Test Type Match (e.g., Stroop)
        if file_test != TARGET_TEST:
            continue

        filepath = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(filepath)

        if EEG_COLUMN not in df.columns or "poorSignal" not in df.columns:
            continue

        raw_volts = df[EEG_COLUMN].values
        poor_signal = df["poorSignal"].values

        # 1. Settling Logic: Skip the noisy 'warm-up' period
        samples_to_discard = DISCARD_SECONDS * ORIGINAL_FS
        if len(raw_volts) <= samples_to_discard:
            continue

        raw_volts = raw_volts[samples_to_discard:]
        poor_signal = poor_signal[samples_to_discard:]

        # 2. Continuous Preprocessing
        processed_eeg = preprocess_eeg(raw_volts)

        # 3. Optimized Windowing (Removed Redundant Zip Loop)
        window_size_128 = 5 * TARGET_FS
        step_size_128 = TARGET_FS  # 1-second slide
        time_series_data = []

        with torch.no_grad():
            for start_128 in range(
                0, len(processed_eeg) - window_size_128, step_size_128
            ):
                # Map 128Hz index back to 512Hz to check signal quality
                start_512 = start_128 * 4
                window_size_512 = window_size_128 * 4

                signal_window = poor_signal[start_512 : start_512 + window_size_512]
                if np.any(signal_window > 45):
                    continue  # Artifact detected (e.g. blink or electrode shift)

                # Prep window for model (Tokens x Samples)
                window = processed_eeg[start_128 : start_128 + window_size_128]
                reshaped = window.reshape(TOKENS_PER_WINDOW, SAMPLES_PER_TOKEN)

                x = torch.tensor(reshaped, dtype=torch.float32).unsqueeze(0).to(device)

                # Inference: Convert raw logits to probabilities
                outputs = model(x)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]

                # SCALING LOGIC: Convert Binary Class (0 or 1) to Peplau Scale (1.0 to 2.0)
                # This gives us a continuous "intensity" value even from a binary model.
                expected_load = (probs[0].item() * 1.0) + (probs[1].item() * 2.0)

                # Calculate actual timestamp for plotting
                seconds_elapsed = DISCARD_SECONDS + (start_128 // TARGET_FS)

                time_series_data.append(
                    {
                        "seconds_elapsed": seconds_elapsed,
                        "cognitive_load": expected_load,
                    }
                )

        if time_series_data:
            run_df = pd.DataFrame(time_series_data)
            # Smooth the output to make the line chart readable
            run_df["smoothed_load"] = (
                run_df["cognitive_load"].rolling(window=10, min_periods=1).mean()
            )
            raw_data[condition].append(run_df)

    # Combine all sessions for the same condition (e.g., average multiple Music runs)
    aggregated_results = {}
    for cond, dfs in raw_data.items():
        if dfs:
            combined = pd.concat(dfs)
            avg_df = combined.groupby("seconds_elapsed").mean().reset_index()
            avg_df["Condition"] = cond
            aggregated_results[cond] = avg_df

    return aggregated_results


def plot_eeg_dashboard(aggregated_results):
    if not aggregated_results:
        print("❌ Error: No matching data found for this participant/test combination.")
        return

    master_df = pd.concat(aggregated_results.values(), ignore_index=True)
    summary_data = master_df.groupby("Condition")["smoothed_load"].mean().reset_index()

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Neural Load Analysis: {TARGET_PARTICIPANT_REGEX.strip('^$')} ({TARGET_TEST})",
        fontsize=16,
        fontweight="bold",
    )

    # Panel 1: Bar Chart (Global Averages)
    sns.barplot(
        data=summary_data, x="Condition", y="smoothed_load", ax=axes[0], palette="magma"
    )
    axes[0].set_title("Total Avg Cognitive Load")
    axes[0].set_ylabel("Peplau Intensity (1.0-2.0)")
    axes[0].set_ylim(1.0, 2.0)

    # Panel 2: Timeline (Fatigue & Adaptation)
    sns.lineplot(
        data=master_df,
        x="seconds_elapsed",
        y="smoothed_load",
        hue="Condition",
        ax=axes[1],
        linewidth=2.5,
        palette="magma",
    )
    axes[1].set_title("Real-time Load (10s Moving Avg)")
    axes[1].set_xlabel("Seconds Since Start")
    axes[1].set_xlim(DISCARD_SECONDS, master_df["seconds_elapsed"].max())
    axes[1].set_ylim(1.0, 2.0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    print(f"🚀 Initializing Inference for Participant: {TARGET_PARTICIPANT_REGEX}")
    model, device = load_model()
    results = analyze_eeg_files(model, device)
    plot_eeg_dashboard(results)

