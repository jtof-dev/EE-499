import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from scipy import signal

# import SingLEM from git submodule
sys.path.append(os.path.join(os.path.dirname(__file__), "SingLEM"))
from SingLEM.model import Config, EEGEncoder

# configuration
DATA_DIR = "data/level_2"
MODEL_WEIGHTS_PATH = "singlem_binary_head.pth"
PRETRAINED_WEIGHTS_PATH = "SingLEM/weights/singlem_pretrained.pt"
TARGET_TEST = "Stroop"

CONDITIONS = ["Silent", "WhiteNoise", "Music", "MusicNL"]
EEG_COLUMN = "eegRawValueVolts"
TARGET_PARTICIPANT_REGEX = r"^Andy$"

# DSP constants
ORIGINAL_FS = 512
TARGET_FS = 128
WINDOW_SECONDS = 5
TOKENS_PER_WINDOW = WINDOW_SECONDS
SAMPLES_PER_TOKEN = TARGET_FS
DISCARD_SECONDS = 60

FILENAME_REGEX = re.compile(
    r"^\d{8}_\d{4}_(?P<participant>[A-Za-z0-9]+)_EEG_(?P<test>Stroop|Typing)_(?P<condition>Silent|WhiteNoise|Music|MusicNL)\.csv$",
    re.IGNORECASE,
)


def preprocess_eeg(raw_volts):
    b, a = signal.butter(4, [0.5, 50.0], btype="bandpass", fs=ORIGINAL_FS)
    filtered = signal.filtfilt(b, a, raw_volts)
    b_notch, a_notch = signal.iirnotch(60.0, 30.0, fs=ORIGINAL_FS)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)
    resampled = signal.decimate(filtered, q=4)
    scaled = resampled * 1e4
    return scaled


class BinaryAnxietyClassifier(nn.Module):
    def __init__(self, unfreeze_last_n=0):
        super().__init__()
        config = Config()
        config.mask_prob = 0.0
        self.feature_extractor = EEGEncoder(config)

        if os.path.exists(PRETRAINED_WEIGHTS_PATH):
            self.feature_extractor.load_state_dict(
                torch.load(
                    PRETRAINED_WEIGHTS_PATH, map_location="cpu", weights_only=True
                )
            )

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        if unfreeze_last_n > 0:
            params = list(self.feature_extractor.parameters())
            for p in params[-unfreeze_last_n:]:
                p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(5 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        features, _, _ = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryAnxietyClassifier(unfreeze_last_n=0)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def extract_run_stats(model, device):
    """runs inference and calculates the mean predicted anxiety for each run."""
    run_stats = []
    participant_filter = re.compile(TARGET_PARTICIPANT_REGEX)

    print(
        f"extracting neural features for {TARGET_PARTICIPANT_REGEX.strip('^$').lower()} ({TARGET_TEST.lower()})..."
    )

    for filename in os.listdir(DATA_DIR):
        match = FILENAME_REGEX.match(filename)
        if not match:
            continue

        file_participant = match.group("participant")
        file_test = match.group("test").capitalize()
        condition = match.group("condition")

        if not participant_filter.search(file_participant) or file_test != TARGET_TEST:
            continue

        filepath = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(filepath)

        if EEG_COLUMN not in df.columns or "poorSignal" not in df.columns:
            continue

        raw_volts = df[EEG_COLUMN].values
        poor_signal = df["poorSignal"].values
        samples_to_discard = DISCARD_SECONDS * ORIGINAL_FS

        if len(raw_volts) <= samples_to_discard:
            continue

        raw_volts = raw_volts[samples_to_discard:]
        poor_signal = poor_signal[samples_to_discard:]
        processed_eeg = preprocess_eeg(raw_volts)

        window_size_128 = 5 * TARGET_FS
        step_size_128 = TARGET_FS
        load_predictions = []

        with torch.no_grad():
            for start_128 in range(
                0, len(processed_eeg) - window_size_128, step_size_128
            ):
                start_512 = start_128 * 4
                window_size_512 = window_size_128 * 4

                signal_window = poor_signal[start_512 : start_512 + window_size_512]
                if np.any(signal_window > 45):
                    continue

                window = processed_eeg[start_128 : start_128 + window_size_128]
                reshaped = window.reshape(TOKENS_PER_WINDOW, SAMPLES_PER_TOKEN)
                x = torch.tensor(reshaped, dtype=torch.float32).unsqueeze(0).to(device)

                outputs = model(x)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                expected_load = (probs[0].item() * 1.0) + (probs[1].item() * 2.0)
                load_predictions.append(expected_load)

        if load_predictions:
            run_stats.append(
                {"Condition": condition, "Mean_Anxiety": np.mean(load_predictions)}
            )

    return pd.DataFrame(run_stats)


if __name__ == "__main__":
    model, device = load_model()
    df_stats = extract_run_stats(model, device)

    if df_stats.empty:
        print("no valid EEG data found.")
        exit()

    print("\nEEG predicted anxiety: statistical analysis")

    # shapiro-wilk test (normality)
    print("\nshapiro-wilk test (normality)")
    all_normal = True
    for cond in CONDITIONS:
        cond_data = df_stats[df_stats["Condition"] == cond]["Mean_Anxiety"]
        if len(cond_data) >= 3:
            stat, p_value = stats.shapiro(cond_data)
            is_normal = "normal" if p_value > 0.05 else "not normal"
            if p_value <= 0.05:
                all_normal = False
            print(
                f"  {cond.lower().ljust(12)}: {is_normal} (p = {p_value:.4f}, n = {len(cond_data)})"
            )
        else:
            print(f"  {cond.lower().ljust(12)}: not enough data (n={len(cond_data)})")
            all_normal = False

    # levene's test (variance)
    print("\nlevene's test (equal variance)")
    groups = [
        df_stats[df_stats["Condition"] == cond]["Mean_Anxiety"].values
        for cond in CONDITIONS
        if len(df_stats[df_stats["Condition"] == cond]) > 0
    ]

    equal_variance = False
    if len(groups) > 1:
        stat, p_value = stats.levene(*groups)
        equal_variance = p_value > 0.05
        is_equal = "equal variances" if equal_variance else "unequal variances"
        print(f"  result      : {is_equal} (p = {p_value:.4f})")

    # main test (ANOVA or kruskal-wallis)
    print("\nmain significance test")
    if all_normal and equal_variance:
        print("  method: one-way ANOVA (data passed all pre-flight checks)")
        f_stat, p_main = stats.f_oneway(*groups)
    else:
        print("  method: kruskal-wallis (data failed normality or variance checks)")
        h_stat, p_main = stats.kruskal(*groups)

    if p_main < 0.05:
        print(f"  significant difference found (p = {p_main:.4f})")
        print(
            "  the model detected a significant shift in predicted anxiety across audio conditions."
        )

        # only run tukey if it was an ANOVA
        if all_normal and equal_variance:
            print("\ntukey HSD post-hoc results")
            active_conditions = [
                cond
                for cond in CONDITIONS
                if len(df_stats[df_stats["Condition"] == cond]) > 0
            ]
            print(f"condition order: {[c.lower() for c in active_conditions]}")
            try:
                print(stats.tukey_hsd(*groups))
            except AttributeError:
                print(
                    "  note: update scipy to >=1.8.0 to view tukey HSD results automatically."
                )
    else:
        print(f"  no significant difference (p = {p_main:.4f})")
        print(
            "  the model's predicted anxiety levels did not differ significantly between audio conditions."
        )
