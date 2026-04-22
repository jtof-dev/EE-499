import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- CONFIGURATION ---
DATA_DIR = "data/level_2"
CONDITIONS = ["Silent", "WhiteNoise", "Music"]
DISCARD_SECONDS = 60  # MUST match the EEG script exactly to ensure time-alignment
TARGET_PARTICIPANT_REGEX = r"^Andy$"  # Strict matching for 'Andy' only

# --- PREPARATION ---
processed_dfs = []
summary_data = []
# Dictionary to store dataframes grouped by condition for cleaner aggregation
data_buckets = {cond: [] for cond in CONDITIONS}

# Compile regex for performance
regex_pattern = re.compile(TARGET_PARTICIPANT_REGEX)

print(
    f"Searching for Metrics data for participant matching: {TARGET_PARTICIPANT_REGEX}"
)

# --- 1. OPTIMIZED DATA LOADING ---
# We iterate through the directory ONCE and sort files into buckets
for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    # Convention: YYYYMMDD_HHMM_PARTICIPANT_DATATYPE_TEST_TESTCONDITION.csv
    parts = file.replace(".csv", "").split("_")

    if len(parts) < 6:
        continue

    participant = parts[2]
    datatype = parts[3]
    test_type = parts[4]
    condition = parts[5]

    # Apply strict filters: Participant, Type (Metrics), and Task (Stroop)
    if not regex_pattern.search(participant):
        continue
    if datatype != "Metrics" or test_type != "Stroop":
        continue
    if condition not in CONDITIONS:
        continue

    # Load and process file
    file_path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(file_path)

    # Calculate relative time in seconds from the start of the recording
    # We round to the nearest second to allow for easy grouping/averaging across runs
    df["seconds_elapsed"] = (
        ((df["timestampMs"] - df["timestampMs"].iloc[0]) / 1000).round().astype(int)
    )

    # --- 2. SYNCHRONIZATION (SETTLING TIME) ---
    # Slice off the first minute to match the EEG preprocessing discard period
    df = df[df["seconds_elapsed"] >= DISCARD_SECONDS].copy()

    if df.empty:
        continue

    # Feature Engineering:
    # Calculate keys per second (10s rolling average) and running error count
    df["keys_per_sec"] = df["keys_pressed"].rolling(window=10, min_periods=1).mean()
    df["cumulative_errors"] = df["errors"].cumsum()

    data_buckets[condition].append(df)

# --- 3. AGGREGATION & SUMMARY ---
for cond, dfs in data_buckets.items():
    if not dfs:
        continue

    # Combine all runs for this specific condition
    combined = pd.concat(dfs)

    # Create an 'Average Session' by grouping by the elapsed second
    avg_df = combined.groupby("seconds_elapsed").mean().reset_index()
    avg_df["Condition"] = cond
    processed_dfs.append(avg_df)

    # Summary Stats: Derived from the total volume of work in this condition
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

# --- 4. DASHBOARD GENERATION ---
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Participant Analysis: {TARGET_PARTICIPANT_REGEX.strip('^$')}",
    fontsize=16,
    fontweight="bold",
)

# Panel 1: Efficiency (Dual Axis - Volume vs. Accuracy)
ax1 = axes[0]
ax2 = ax1.twinx()  # Overlay accuracy line on top of volume bars

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
ax2.set_ylim(80, 105)  # Keep accuracy focused on the high-performance range
ax2.grid(False)  # Clean up overlapping grid lines

# Panel 2: Throughput over Time
sns.lineplot(
    data=master_df, x="seconds_elapsed", y="keys_per_sec", hue="Condition", ax=axes[1]
)
axes[1].set_title("Cognitive Throughput (10s Rolling Avg)")
axes[1].set_xlim(DISCARD_SECONDS, master_df["seconds_elapsed"].max())

# Panel 3: Error Accumulation
sns.lineplot(
    data=master_df,
    x="seconds_elapsed",
    y="cumulative_errors",
    hue="Condition",
    ax=axes[2],
)
axes[2].set_title("Error Accumulation Over Time")
axes[2].set_xlim(DISCARD_SECONDS, master_df["seconds_elapsed"].max())

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
plt.show()

