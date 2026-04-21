import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_DIR = "data/level_2"
CONDITIONS = ["Silent", "WhiteNoise", "Music"]

# load and aggregate data
processed_dfs = []
summary_data = []

for cond in CONDITIONS:
    cond_dfs = []

    # simple string match for files
    for file in os.listdir(DATA_DIR):
        if "Stroop" in file and cond in file:
            df = pd.read_csv(os.path.join(DATA_DIR, file))

            # normalize time and calculate rolling/cumulative metrics
            df["seconds_elapsed"] = (
                (df["timestampMs"] - df["timestampMs"].iloc[0]) / 1000
            ).round()
            df["keys_per_sec"] = (
                df["keys_pressed"].rolling(window=10, min_periods=1).mean()
            )
            df["cumulative_errors"] = df["errors"].cumsum()

            cond_dfs.append(df)

    # average the runs for this condition
    if cond_dfs:
        combined = pd.concat(cond_dfs)
        avg_df = combined.groupby("seconds_elapsed").mean().reset_index()
        avg_df["Condition"] = cond
        processed_dfs.append(avg_df)

        # summary stats for the bar chart
        total_keys = combined["keys_pressed"].sum()
        total_errors = combined["errors"].sum()
        summary_data.append(
            {
                "Condition": cond,
                "Avg Attempts per Run": total_keys / len(cond_dfs),
                "Accuracy %": (total_keys - total_errors) / total_keys * 100,
            }
        )

master_df = pd.concat(processed_dfs)
summary_df = pd.DataFrame(summary_data)

# generate dashboard
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Stroop Test: Audio Distraction Analysis", fontsize=16, fontweight="bold")

# panel 1: efficiency
ax1 = axes[0]
ax2 = ax1.twinx()
sns.barplot(
    data=summary_df, x="Condition", y="Avg Attempts per Run", ax=ax1, color="skyblue"
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
ax1.set_title("Global Efficiency")
ax2.set_ylim(80, 100)

# panel 2: throughput over time
sns.lineplot(
    data=master_df, x="seconds_elapsed", y="keys_per_sec", hue="Condition", ax=axes[1]
)
axes[1].set_title("Cognitive Throughput (10s Avg)")
axes[1].set_xlim(0, 300)

# panel 3: error accumulation
sns.lineplot(
    data=master_df,
    x="seconds_elapsed",
    y="cumulative_errors",
    hue="Condition",
    ax=axes[2],
)
axes[2].set_title("Cumulative Errors Over Time")
axes[2].set_xlim(0, 300)

plt.tight_layout()
plt.show()
