import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# configuration
DATA_DIR = "data/level_2"
CONDITIONS = ["Silent", "WhiteNoise", "Music", "MusicNL"]
DISCARD_SECONDS = 60  # must match the EEG script
TARGET_PARTICIPANT_REGEX = r"^Andy$"

# preparation
processed_dfs = []
summary_data = []
run_stats = []  # store individual run statistics for chronological plots
data_buckets = {cond: [] for cond in CONDITIONS}

# Compile regex for performance
regex_pattern = re.compile(TARGET_PARTICIPANT_REGEX)

print(
    f"Searching for Metrics data for participant matching: '{TARGET_PARTICIPANT_REGEX}', Test: 'Typing'"
)

# iterate through the directory and sort files into buckets
for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    # convention: YYYYMMDD_HHMM_PARTICIPANT_DATATYPE_TEST_TESTCONDITION.csv
    parts = file.replace(".csv", "").split("_")

    if len(parts) < 6:
        continue

    # Parse the timestamp from the file name
    timestamp_str = parts[0] + "_" + parts[1]
    try:
        run_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
    except ValueError:
        continue

    participant = parts[2]
    datatype = parts[3]
    test_type = parts[4]
    condition = parts[5]

    if not regex_pattern.search(participant):
        continue
    if datatype != "Metrics" or test_type != "Typing":
        continue
    if condition not in CONDITIONS:
        continue

    file_path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(file_path)

    # 1. Limit errors to maximum 1 per second (accounts for tracking oversight)
    df["errors"] = df["errors"].clip(upper=1)

    # 2. The total_correct column is cumulative. diff() gives us words completed per second.
    df["words_completed"] = df["total_words"].diff().fillna(0)

    # calculate relative time in seconds from the start of the recording
    df["seconds_elapsed"] = (
        ((df["timestampMs"] - df["timestampMs"].iloc[0]) / 1000).round().astype(int)
    )

    # synchronization and settling time
    df = df[df["seconds_elapsed"] >= DISCARD_SECONDS].copy()

    if df.empty:
        continue

    # 10s rolling average for words per second
    df["words_per_sec"] = df["words_completed"].rolling(window=10, min_periods=1).mean()
    df["cumulative_errors"] = df["errors"].cumsum()

    data_buckets[condition].append(df)

    # --- Calculate stats per specific run for chronological tracking ---
    total_correct_run = df["words_completed"].sum()
    total_errors_run = df["errors"].sum()

    # 3. Accuracy based on total words vs total mistakes
    total_attempts_run = total_correct_run + total_errors_run
    if total_attempts_run > 0:
        accuracy_run = (total_correct_run / total_attempts_run) * 100
    else:
        accuracy_run = 0

    run_stats.append(
        {
            "Datetime": run_datetime,
            "Condition": condition,
            "Accuracy %": accuracy_run,
            "Avg Words/Sec": df["words_per_sec"].mean(),
            "Total Correct Words": total_correct_run,
        }
    )

# aggregation and summary
for cond, dfs in data_buckets.items():
    if not dfs:
        continue

    combined = pd.concat(dfs)

    avg_df = combined.groupby("seconds_elapsed").mean().reset_index()
    avg_df["Condition"] = cond
    processed_dfs.append(avg_df)

    total_correct = combined["words_completed"].sum()
    total_errors = combined["errors"].sum()

    total_attempts = total_correct + total_errors
    if total_attempts > 0:
        accuracy = (total_correct / total_attempts) * 100
    else:
        accuracy = 0

    summary_data.append(
        {
            "Condition": cond,
            "Avg Correct Words per Run": total_correct / len(dfs),
            "Accuracy %": accuracy,
        }
    )

if not processed_dfs:
    print("No matching data found. Please check your DATA_DIR and Regex.")
    exit()

master_df = pd.concat(processed_dfs)
summary_df = pd.DataFrame(summary_data)

# Sort strictly by time, then convert to a string to remove massive X-axis time gaps
run_stats_df = pd.DataFrame(run_stats).sort_values("Datetime")
run_stats_df["Run_Label"] = run_stats_df["Datetime"].dt.strftime("%m/%d %H:%M")


# dashboard generation
sns.set_theme(style="darkgrid")
participant_name = TARGET_PARTICIPANT_REGEX.strip("^$")

# ---------------------------------------------------------
# WINDOW 1: Output & Efficiency (Bar + Line Overlay)
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 6))
fig1.canvas.manager.set_window_title("Typing - Efficiency Summary")

sns.barplot(
    data=summary_df,
    x="Condition",
    y="Avg Correct Words per Run",
    hue="Condition",
    ax=ax1,
    palette="magma",
    alpha=0.8,
    legend=False,
)

ax2 = ax1.twinx()
sns.lineplot(
    data=summary_df,
    x="Condition",
    y="Accuracy %",
    ax=ax2,
    color="black",
    marker="o",
    linewidth=2.5,
)

# Add explicit percentage text labels directly above the dots
for i, acc in enumerate(summary_df["Accuracy %"]):
    ax2.text(
        i,
        acc + 0.3,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        color="black",
        fontweight="bold",
    )

ax1.set_title(
    f"Total Output (Typing) (Post-{DISCARD_SECONDS}s)\nParticipant: {participant_name}"
)
ax1.set_xlabel("Test Condition")
ax1.set_ylabel("Avg Total Correct Words")
ax2.set_ylabel("Accuracy (%)")

# STRICTLY lock the Y-axis ceiling so the graph visually makes sense
y_min = min(80, summary_df["Accuracy %"].min() - 5)
ax2.set_ylim(y_min, 101)  # 101 allows a tiny bit of padding for the text labels
ax2.grid(False)
plt.tight_layout()

# ---------------------------------------------------------
# WINDOW 2: Cognitive Throughput
# ---------------------------------------------------------
fig2, ax_thr = plt.subplots(figsize=(8, 6))
fig2.canvas.manager.set_window_title("Typing - Cognitive Throughput")

sns.lineplot(
    data=master_df,
    x="seconds_elapsed",
    y="words_per_sec",
    hue="Condition",
    ax=ax_thr,
    palette="magma",
    linewidth=2,
)
ax_thr.set_xlabel("Seconds Elapsed")
ax_thr.set_ylabel("Words per Second")
ax_thr.set_title(
    f"Cognitive Throughput (Typing) (10s Rolling Avg) - {participant_name}"
)
ax_thr.set_xlim(DISCARD_SECONDS, master_df["seconds_elapsed"].max())
plt.tight_layout()

# ---------------------------------------------------------
# WINDOW 3: Error Accumulation
# ---------------------------------------------------------
fig3, ax_err = plt.subplots(figsize=(8, 6))
fig3.canvas.manager.set_window_title("Typing - Error Accumulation")

sns.lineplot(
    data=master_df,
    x="seconds_elapsed",
    y="cumulative_errors",
    hue="Condition",
    ax=ax_err,
    palette="magma",
    linewidth=2,
)
ax_err.set_xlabel("Seconds Elapsed")
ax_err.set_ylabel("Cumulative Errors")
ax_err.set_title(f"Error Accumulation Over Time (Typing) - {participant_name}")
ax_err.set_xlim(DISCARD_SECONDS, master_df["seconds_elapsed"].max())
plt.tight_layout()

# ---------------------------------------------------------
# WINDOW 4: Chronological Improvement (Accuracy)
# ---------------------------------------------------------
fig4, ax_chron_acc = plt.subplots(figsize=(10, 6))
fig4.canvas.manager.set_window_title("Typing - Chronological Accuracy")

sns.lineplot(
    data=run_stats_df,
    x="Run_Label",
    y="Accuracy %",
    hue="Condition",
    marker="o",
    palette="magma",
    linewidth=2,
    ax=ax_chron_acc,
)
ax_chron_acc.set_title(
    f"Chronological Performance Improvement (Accuracy - Typing) - {participant_name}"
)
ax_chron_acc.set_xlabel("Run Timestamp")
ax_chron_acc.set_ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.tight_layout()

# ---------------------------------------------------------
# WINDOW 5: Chronological Improvement (Throughput)
# ---------------------------------------------------------
fig5, ax_chron_kps = plt.subplots(figsize=(10, 6))
fig5.canvas.manager.set_window_title("Typing - Chronological Speed")

sns.lineplot(
    data=run_stats_df,
    x="Run_Label",
    y="Avg Words/Sec",
    hue="Condition",
    marker="o",
    palette="magma",
    linewidth=2,
    ax=ax_chron_kps,
)
ax_chron_kps.set_title(
    f"Chronological Performance Improvement (Speed - Typing) - {participant_name}"
)
ax_chron_kps.set_xlabel("Run Timestamp")
ax_chron_kps.set_ylabel("Avg Words per Second")
plt.xticks(rotation=45)
plt.tight_layout()

# Show all generated popup windows simultaneously
plt.show()
