import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# configuration
DATA_DIR = "data/level_2"
CONDITIONS = ["Silent", "WhiteNoise", "Music"]
DISCARD_SECONDS = 60  # must match the EEG script
TARGET_PARTICIPANT_REGEX = r"^Andy$"

# preparation
processed_dfs = []
summary_data = []
# dictionary to store dataframes grouped by condition
data_buckets = {cond: [] for cond in CONDITIONS}

# Compile regex for performance
regex_pattern = re.compile(TARGET_PARTICIPANT_REGEX)

print(
    f"Searching for Metrics data for participant matching: {TARGET_PARTICIPANT_REGEX}"
)

# iterate through the directory and sort files into buckets
for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    # convention: YYYYMMDD_HHMM_PARTICIPANT_DATATYPE_TEST_TESTCONDITION.csv
    parts = file.replace(".csv", "").split("_")

    if len(parts) < 6:
        continue

    participant = parts[2]
    datatype = parts[3]
    test_type = parts[4]
    condition = parts[5]

    # apply strict filters: participant, type, and task
    if not regex_pattern.search(participant):
        continue
    if datatype != "Metrics" or test_type != "Stroop":
        continue
    if condition not in CONDITIONS:
        continue

    # load and process file
    file_path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(file_path)

    # calculate relative time in seconds from the start of the recording
    # round to the nearest second to allow for easy grouping/averaging across runs
    df["seconds_elapsed"] = (
        ((df["timestampMs"] - df["timestampMs"].iloc[0]) / 1000).round().astype(int)
    )

    # synchronization and settling time
    df = df[df["seconds_elapsed"] >= DISCARD_SECONDS].copy()

    if df.empty:
        continue

    # calculate keys per second (10s rolling average) and running error count
    df["keys_per_sec"] = df["keys_pressed"].rolling(window=10, min_periods=1).mean()
    df["cumulative_errors"] = df["errors"].cumsum()

    data_buckets[condition].append(df)

# aggregation and summary
for cond, dfs in data_buckets.items():
    if not dfs:
        continue

    # combine all runs for this specific condition
    combined = pd.concat(dfs)

    # create an average session by grouping by the elapsed second
    avg_df = combined.groupby("seconds_elapsed").mean().reset_index()
    avg_df["Condition"] = cond
    processed_dfs.append(avg_df)

    # summary stats derived from the total volume of work in this condition
    total_keys = combined["keys_pressed"].sum()
    total_errors = combined["errors"].sum()

    summary_data.append(
        {
            "Condition": cond,
            "Avg Attempts per Run": total_keys / len(dfs),
            "Accuracy %": (
                ((total_keys - total_errors) / total_keys * 100)
                if total_keys > 0
                else 0
            ),
        }
    )

if not processed_dfs:
    print("No matching data found. Please check your DATA_DIR and Regex.")
    exit()

master_df = pd.concat(processed_dfs)
summary_df = pd.DataFrame(summary_data)

# dashboard generation
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Participant Analysis: {TARGET_PARTICIPANT_REGEX.strip('^$')}",
    fontsize=16,
    fontweight="bold",
)

# panel 1: efficiency
ax1 = axes[0]
ax2 = ax1.twinx()  # overlay accuracy line on top of volume bars

sns.barplot(
    data=summary_df,
    x="Condition",
    y="Avg Attempts per Run",
    ax=ax1,
    color="skyblue",
    alpha=0.7,
)
sns.lineplot(
    data=summary_df,
    x="Condition",
    y="Accuracy %",
    ax=ax2,
    color="red",
    marker="o",
    linewidth=2,
)

ax1.set_title(f"Total Output (Post-{DISCARD_SECONDS}s)")
ax2.set_ylim(80, 105)
ax2.grid(False)

# panel 2: throughput over time
sns.lineplot(
    data=master_df, x="seconds_elapsed", y="keys_per_sec", hue="Condition", ax=axes[1]
)
axes[1].set_title("Cognitive Throughput (10s Rolling Avg)")
axes[1].set_xlim(DISCARD_SECONDS, master_df["seconds_elapsed"].max())

# panel 3: error accumulation
sns.lineplot(
    data=master_df,
    x="seconds_elapsed",
    y="cumulative_errors",
    hue="Condition",
    ax=axes[2],
)
axes[2].set_title("Error Accumulation Over Time")
axes[2].set_xlim(DISCARD_SECONDS, master_df["seconds_elapsed"].max())

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
