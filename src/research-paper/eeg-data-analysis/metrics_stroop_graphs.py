import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# configuration
DATA_DIR = "data/level_2"
CONDITIONS = ["Silent", "WhiteNoise", "Music", "MusicNL"]
DISCARD_SECONDS = 60  # for plotting
TARGET_PARTICIPANT_REGEX = r"^Andy$"

# preparation for plotting
processed_dfs = []
summary_data = []
run_stats = []
data_buckets_plot = {cond: [] for cond in CONDITIONS}

# preparation for CLI table (math from metrics_stroop_table.py)
condition_stats_table = {cond: [] for cond in CONDITIONS}

# compile regex for performance
regex_pattern = re.compile(TARGET_PARTICIPANT_REGEX)

print(
    f"processing metrics for participant matching: {TARGET_PARTICIPANT_REGEX.lower()}\n"
)

# iterate through the directory
for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    parts = file.replace(".csv", "").split("_")
    if len(parts) < 6:
        continue

    # parse the timestamp
    timestamp_str = parts[0] + "_" + parts[1]
    try:
        run_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
    except ValueError:
        continue

    participant, datatype, test_type, condition = parts[2], parts[3], parts[4], parts[5]

    if (
        not regex_pattern.search(participant)
        or datatype != "Metrics"
        or test_type != "Stroop"
        or condition not in CONDITIONS
    ):
        continue

    file_path = os.path.join(DATA_DIR, file)
    df_raw = pd.read_csv(file_path)

    if df_raw.empty:
        continue

    # table math (logic from metrics_stroop_table.py)
    # uses full duration of run without discarding start
    total_keys_tbl = df_raw["keys_pressed"].sum()
    total_errors_tbl = df_raw["errors"].sum()
    total_correct_tbl = df_raw["total_correct"].iloc[-1]
    duration_sec_tbl = (
        df_raw["timestampMs"].iloc[-1] - df_raw["timestampMs"].iloc[0]
    ) / 1000.0

    condition_stats_table[condition].append(
        {
            "keys": total_keys_tbl,
            "correct": total_correct_tbl,
            "errors": total_errors_tbl,
            "duration": duration_sec_tbl,
        }
    )

    # plot math (logic from metrics_stroop_analysis.py)
    # discards first 60 seconds for synchronization
    df_plot = df_raw.copy()
    df_plot["seconds_elapsed"] = (
        ((df_plot["timestampMs"] - df_plot["timestampMs"].iloc[0]) / 1000)
        .round()
        .astype(int)
    )
    df_plot = df_plot[df_plot["seconds_elapsed"] >= DISCARD_SECONDS].copy()

    if not df_plot.empty:
        df_plot["keys_per_sec"] = (
            df_plot["keys_pressed"].rolling(window=10, min_periods=1).mean()
        )
        df_plot["cumulative_errors"] = df_plot["errors"].cumsum()
        data_buckets_plot[condition].append(df_plot)

        total_keys_run = df_plot["keys_pressed"].sum()
        total_errors_run = df_plot["errors"].sum()
        correct_run = max(0, total_keys_run - total_errors_run)
        accuracy_run = (correct_run / total_keys_run * 100) if total_keys_run > 0 else 0

        run_stats.append(
            {
                "Datetime": run_datetime,
                "Condition": condition,
                "Accuracy %": accuracy_run,
                "Avg Keys/Sec": df_plot["keys_per_sec"].mean(),
                "Total Attempts": total_keys_run,
            }
        )

# print CLI table
print(
    f"{'condition':<12} | {'runs':<5} | {'avg keys':<10} | {'avg correct':<12} | {'avg errors':<10} | {'keys/sec':<10} | {'accuracy %':<10}"
)
for cond in CONDITIONS:
    runs = condition_stats_table[cond]
    num_runs = len(runs)
    if num_runs == 0:
        print(
            f"{cond.lower():<12} | {num_runs:<5} | {'-':<10} | {'-':<12} | {'-':<10} | {'-':<10} | {'-':<10}"
        )
        continue

    sum_keys = sum(r["keys"] for r in runs)
    sum_correct = sum(r["correct"] for r in runs)
    sum_errors = sum(r["errors"] for r in runs)
    sum_duration = sum(r["duration"] for r in runs)

    avg_keys = sum_keys / num_runs
    avg_correct = sum_correct / num_runs
    avg_errors = sum_errors / num_runs
    overall_kps = sum_keys / sum_duration if sum_duration > 0 else 0
    overall_acc = (sum_correct / sum_keys * 100) if sum_keys > 0 else 0

    print(
        f"{cond.lower():<12} | {num_runs:<5} | {avg_keys:<10.1f} | {avg_correct:<12.1f} | {avg_errors:<10.1f} | {overall_kps:<10.2f} | {overall_acc:<10.2f}"
    )

# map labels for academic formatting
condition_map = {
    "Silent": "Silent (Control)",
    "WhiteNoise": "White Noise",
    "Music": "Lyrical Music",
    "MusicNL": "Non-Lyrical Music",
}

# aggregation and plotting prep
for cond, dfs in data_buckets_plot.items():
    if not dfs:
        continue
    combined = pd.concat(dfs)
    avg_df = combined.groupby("seconds_elapsed").mean().reset_index()
    avg_df["Condition"] = condition_map.get(cond, cond)
    processed_dfs.append(avg_df)

    total_keys = combined["keys_pressed"].sum()
    total_errors = combined["errors"].sum()
    accuracy = (
        (max(0, total_keys - total_errors) / total_keys * 100) if total_keys > 0 else 0
    )
    summary_data.append(
        {
            "Condition": condition_map.get(cond, cond),
            "Avg Attempts per Run": total_keys / len(dfs),
            "Accuracy %": accuracy,
        }
    )

if not processed_dfs:
    print("\nno matching data found for plots. check directory/regex.")
    exit()

master_df = pd.concat(processed_dfs)
summary_df = pd.DataFrame(summary_data)
run_stats_df = pd.DataFrame(run_stats).sort_values("Datetime")
run_stats_df["Condition"] = run_stats_df["Condition"].map(condition_map)
run_stats_df["Run_Label"] = run_stats_df["Datetime"].dt.strftime("%m/%d %H:%M")
participant_name = TARGET_PARTICIPANT_REGEX.strip("^$").capitalize()

# graph formatting
sns.set_theme(style="darkgrid")

# fig 1: efficiency summary
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(
    data=summary_df,
    x="Condition",
    y="Avg Attempts per Run",
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
    f"Mean Total Output and Task Accuracy: {participant_name}",
    fontsize=14,
    fontweight="bold",
)
ax1.set_xlabel("Auditory Environmental Condition", fontsize=12)
ax1.set_ylabel("Average Total Attempts per Session", fontsize=12)
ax2.set_ylabel("Mean Accuracy Percentage (%)", fontsize=12)
ax2.set_ylim(min(80, summary_df["Accuracy %"].min() - 5), 101)
ax2.grid(False)
plt.tight_layout()

# fig 2: cognitive throughput
fig2, ax_thr = plt.subplots(figsize=(8, 6))
sns.lineplot(
    data=master_df,
    x="seconds_elapsed",
    y="keys_per_sec",
    hue="Condition",
    ax=ax_thr,
    palette="magma",
    linewidth=2,
)
ax_thr.set_title(
    f"Real-Time Cognitive Throughput: {participant_name}",
    fontsize=14,
    fontweight="bold",
)
ax_thr.set_xlabel("Time Elapsed (Seconds)", fontsize=12)
ax_thr.set_ylabel("Keys Pressed per Second (10s Rolling Average)", fontsize=12)
ax_thr.set_xlim(DISCARD_SECONDS, master_df["seconds_elapsed"].max())
ax_thr.legend(title="Auditory Condition", loc="lower right", frameon=True)
plt.tight_layout()

# fig 3: error accumulation
fig3, ax_err = plt.subplots(figsize=(8, 6))
sns.lineplot(
    data=master_df,
    x="seconds_elapsed",
    y="cumulative_errors",
    hue="Condition",
    ax=ax_err,
    palette="magma",
    linewidth=2,
)
ax_err.set_title(
    f"Cumulative Error Rate Over Time: {participant_name}",
    fontsize=14,
    fontweight="bold",
)
ax_err.set_xlabel("Time Elapsed (Seconds)", fontsize=12)
ax_err.set_ylabel("Total Cumulative Errors", fontsize=12)
ax_err.set_xlim(DISCARD_SECONDS, master_df["seconds_elapsed"].max())
ax_err.legend(title="Auditory Condition", loc="lower right", frameon=True)
plt.tight_layout()

# fig 4: chronological improvement (accuracy)
fig4, ax_chron_acc = plt.subplots(figsize=(10, 6))
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
    f"Longitudinal Accuracy Performance by Session: {participant_name}",
    fontsize=14,
    fontweight="bold",
)
ax_chron_acc.set_xlabel("Session Timestamp", fontsize=12)
ax_chron_acc.set_ylabel("Accuracy Percentage (%)", fontsize=12)
ax_chron_acc.legend(title="Auditory Condition", loc="lower right", frameon=True)
plt.xticks(rotation=45)
plt.tight_layout()

# fig 5: chronological improvement (speed)
fig5, ax_chron_kps = plt.subplots(figsize=(10, 6))
sns.lineplot(
    data=run_stats_df,
    x="Run_Label",
    y="Avg Keys/Sec",
    hue="Condition",
    marker="o",
    palette="magma",
    linewidth=2,
    ax=ax_chron_kps,
)
ax_chron_kps.set_title(
    f"Longitudinal Speed Performance by Session: {participant_name}",
    fontsize=14,
    fontweight="bold",
)
ax_chron_kps.set_xlabel("Session Timestamp", fontsize=12)
ax_chron_kps.set_ylabel("Average Keys per Second", fontsize=12)
ax_chron_kps.legend(title="Auditory Condition", loc="lower right", frameon=True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

