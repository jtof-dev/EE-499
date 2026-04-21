import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

# --- CONFIGURATION ---
DATA_DIR = "data/level_2"
MODEL_WEIGHTS_PATH = "singlem_anxiety_head.pth"
TARGET_TEST = "Stroop"  # Change to "Typing" to analyze the other task
CONDITIONS = ["Silent", "WhiteNoise", "Music"]
EEG_COLUMN = "EXG Channel 0"
SAMPLE_RATE = 512
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS
STEP_SIZE = SAMPLE_RATE * 1  # Slide by 1 second to get a reading every second

# Regex for EEG files (Notice DATATYPE is hardcoded to 'EEG')
FILENAME_REGEX = re.compile(
    r"^\d{8}_\d{4}_[A-Za-z0-9]+_EEG_(?P<test>Stroop|Typing)_(?P<condition>Silent|WhiteNoise|Music)\.csv$",
    re.IGNORECASE,
)


# --- MODEL ARCHITECTURE ---
# Must match your training script to load the weights properly
class AnxietyClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(AnxietyClassifier, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnxietyClassifier(num_classes=4)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: {MODEL_WEIGHTS_PATH} not found. Using untrained weights.")
    model.to(device)
    model.eval()
    return model, device


def analyze_eeg_files(model, device):
    """Processes EEG files and uses the model to predict cognitive load over time."""
    raw_data = {cond: [] for cond in CONDITIONS}

    for filename in os.listdir(DATA_DIR):
        match = FILENAME_REGEX.match(filename)
        if match and match.group("test").capitalize() == TARGET_TEST:
            condition = match.group("condition")
            if condition not in CONDITIONS:
                continue

            filepath = os.path.join(DATA_DIR, filename)
            df = pd.read_csv(filepath)

            if EEG_COLUMN not in df.columns:
                continue

            raw_eeg = df[EEG_COLUMN].values
            time_series_data = []

            # Slide through the EEG data 1 second at a time
            with torch.no_grad():
                for start_idx in range(0, len(raw_eeg) - WINDOW_SIZE, STEP_SIZE):
                    window = raw_eeg[start_idx : start_idx + WINDOW_SIZE]

                    # Normalize identically to training
                    window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                    x = (
                        torch.tensor(window, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )

                    outputs = model(x)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

                    # Calculate "Expected Cognitive Load" (Continuous scale 0.0 to 3.0)
                    # Instead of a hard 1, 2, or 3, this blends the probabilities.
                    # E.g., if it's 50% Level 1 and 50% Level 2, the score is 1.5.
                    expected_load = sum(
                        i * prob.item() for i, prob in enumerate(probabilities)
                    )

                    seconds_elapsed = start_idx // SAMPLE_RATE
                    time_series_data.append(
                        {
                            "seconds_elapsed": seconds_elapsed,
                            "cognitive_load": expected_load,
                        }
                    )

            if time_series_data:
                run_df = pd.DataFrame(time_series_data)
                # Smooth the data with a 10-second rolling average
                run_df["smoothed_load"] = (
                    run_df["cognitive_load"].rolling(window=10, min_periods=1).mean()
                )
                raw_data[condition].append(run_df)

    # Aggregate runs by condition
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
        print("No valid EEG data found to plot.")
        return

    master_df = pd.concat(aggregated_results.values(), ignore_index=True)

    # Calculate global averages for the bar chart
    summary_data = master_df.groupby("Condition")["smoothed_load"].mean().reset_index()

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{TARGET_TEST} Test: Neural Response to Audio Distractions",
        fontsize=16,
        fontweight="bold",
    )

    # Panel 1: Overall Brain Strain (Bar Chart)
    sns.barplot(
        data=summary_data,
        x="Condition",
        y="smoothed_load",
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title("Average Global Cognitive Load")
    axes[0].set_ylabel("Predicted Peplau Level (0.0 = Relaxed, 3.0 = Panic)")
    axes[0].set_ylim(0, 3.0)

    # Panel 2: Cognitive Load Timeline
    sns.lineplot(
        data=master_df,
        x="seconds_elapsed",
        y="smoothed_load",
        hue="Condition",
        ax=axes[1],
        linewidth=2,
        palette="viridis",
    )
    axes[1].set_title("Neural Fatigue / Adaptation Over Time (10s Avg)")
    axes[1].set_xlabel("Seconds Elapsed")
    axes[1].set_ylabel("Predicted Peplau Level")
    axes[1].set_xlim(0, 300)
    axes[1].set_ylim(0, 3.0)

    # Add subtle horizontal lines to denote Peplau thresholds
    axes[1].axhline(
        1.0, color="gray", linestyle="--", alpha=0.5, label="Level 2 Threshold"
    )
    axes[1].axhline(
        2.0, color="red", linestyle="--", alpha=0.3, label="Level 3 Threshold"
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


if __name__ == "__main__":
    print(f"Loading SingLEM Model...")
    model, device = load_model()

    print(f"Processing '{DATA_DIR}' EEG files...")
    results = analyze_eeg_files(model, device)

    plot_eeg_dashboard(results)
