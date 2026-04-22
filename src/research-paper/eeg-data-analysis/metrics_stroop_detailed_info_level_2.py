import os
import re

import pandas as pd

# configuration
DATA_DIR = "data/level_2"
CONDITIONS = ["Silent", "WhiteNoise", "Music"]
TARGET_PARTICIPANT_REGEX = r"^Andy$"

regex_pattern = re.compile(TARGET_PARTICIPANT_REGEX)

# to store aggregated stats per run
condition_stats = {cond: [] for cond in CONDITIONS}

print(
    f"Aggregating Total Metrics for participant matching: {TARGET_PARTICIPANT_REGEX}\n"
)

# 1. extract and calculate stats for every individual file
for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    parts = file.replace(".csv", "").split("_")
    if len(parts) < 6:
        continue

    participant, datatype, test_type, condition = parts[2], parts[3], parts[4], parts[5]

    if (
        not regex_pattern.search(participant)
        or datatype != "Metrics"
        or test_type != "Stroop"
        or condition not in CONDITIONS
    ):
        continue

    # read data
    file_path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(file_path)

    if df.empty:
        continue

    # calculate totals for the ENTIRE run
    total_keys = df["keys_pressed"].sum()
    total_errors = df["errors"].sum()
    total_correct = df["total_correct"].iloc[
        -1
    ]  # grabbing the final tally is safe again
    duration_sec = (df["timestampMs"].iloc[-1] - df["timestampMs"].iloc[0]) / 1000.0

    # store the run metrics in the bucket
    condition_stats[condition].append(
        {
            "keys": total_keys,
            "correct": total_correct,
            "errors": total_errors,
            "duration": duration_sec,
        }
    )

# 2. average the duplicate runs per condition and print the table
print(
    f"{'Condition':<12} | {'Runs':<5} | {'Avg Keys':<10} | {'Avg Correct':<12} | {'Avg Errors':<10} | {'Keys/Sec':<10} | {'Accuracy %':<10}"
)
print("-" * 85)

for cond in CONDITIONS:
    runs = condition_stats[cond]
    num_runs = len(runs)

    if num_runs == 0:
        print(
            f"{cond:<12} | {num_runs:<5} | {'-':<10} | {'-':<12} | {'-':<10} | {'-':<10} | {'-':<10}"
        )
        continue

    # sum all metrics across the duplicate runs
    sum_keys = sum(r["keys"] for r in runs)
    sum_correct = sum(r["correct"] for r in runs)
    sum_errors = sum(r["errors"] for r in runs)
    sum_duration = sum(r["duration"] for r in runs)

    # calculate averages
    avg_keys = sum_keys / num_runs
    avg_correct = sum_correct / num_runs
    avg_errors = sum_errors / num_runs

    # overall keys per second and accuracy for this condition
    overall_kps = sum_keys / sum_duration if sum_duration > 0 else 0
    overall_acc = (sum_correct / sum_keys * 100) if sum_keys > 0 else 0

    print(
        f"{cond:<12} | {num_runs:<5} | {avg_keys:<10.1f} | {avg_correct:<12.1f} | {avg_errors:<10.1f} | {overall_kps:<10.2f} | {overall_acc:<10.2f}"
    )

print("-" * 85)

