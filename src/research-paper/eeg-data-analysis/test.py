import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def calculate_run_anxiety(file_path):
    """Processes a single EEG file and returns the mean PSS scores based on Saeed et al."""
    df = pd.read_csv(file_path)

    # We only need betaLow and betaHigh for the Saeed calculation
    df_freq = df.drop_duplicates(
        subset=["delta", "theta", "alphaLow", "betaLow", "betaHigh"]
    ).copy()

    # Filter out noisy data based on the headset's built-in metric
    df_clean = df_freq[df_freq["poorSignal"] == 0].copy()

    if df_clean.empty:
        return None

    # Apply the Multiple Linear Regression formula from Saeed et al. (2017)
    # PSS = b0 + b1(beta_L) + b2(beta_H)
    b0 = -2.35
    b1 = 2.29e-6
    b2 = 1.20e-7

    df_clean["saeed_stress_score"] = (
        b0 + (b1 * df_clean["betaLow"]) + (b2 * df_clean["betaHigh"])
    )

    total_beta = df_clean["betaLow"].mean() + df_clean["betaHigh"].mean()
    total_alpha = df_clean["alphaLow"].mean() + df_clean["alphaHigh"].mean()

    return {
        "mean_saeed_stress": df_clean["saeed_stress_score"].mean(),
        "mean_beta_alpha_ratio": total_beta / (total_alpha + 1e-5),
        "clean_epochs": len(df_clean),
    }


def analyze_experiment_runs(data_dir, target_participant, target_test=None):
    print(f"Scanning '{data_dir}' for participant '{target_participant.lower()}'...")

    run_results = []

    for file in os.listdir(data_dir):
        if not file.endswith(".csv"):
            continue

        # Parse based on: yyyymmdd_hhmm_participant_datatype_test_testcondition.csv
        filename_no_ext = file.replace(".csv", "")
        parts = filename_no_ext.split("_")

        # Ensure it matches the expected 6-part structure
        if len(parts) != 6:
            continue

        date, time, participant, datatype, test_type, condition = parts

        # Filter for the correct participant and ensure it's EEG data
        if participant != target_participant or datatype != "EEG":
            continue

        # Optional: filter by specific test
        if target_test and test_type != target_test:
            continue

        file_path = os.path.join(data_dir, file)
        metrics = calculate_run_anxiety(file_path)

        if metrics:
            run_results.append(
                {
                    "Filename": file,
                    "Test_Task": test_type,
                    "Condition": condition,
                    "Mean_Anxiety_Score": metrics["mean_saeed_stress"],
                    "Valid_Epochs": metrics["clean_epochs"],
                }
            )

    if not run_results:
        print("No valid data found to compare.")
        return None

    df_results = pd.DataFrame(run_results)

    # CLI Summary Table
    print("\nAggregated Predicted PSS Levels")
    if target_test:
        print(f"Filtered for task: {target_test.lower()}")

    summary_table = (
        df_results.groupby(["Test_Task", "Condition"])
        .agg(
            Runs=("Filename", "count"),
            Avg_Predicted_PSS=("Mean_Anxiety_Score", "mean"),
        )
        .reset_index()
    )
    summary_table.columns = [c.lower() for c in summary_table.columns]
    print(summary_table.to_string(index=False, float_format="%.3f"))

    # Statistical Testing - Now separates by Test_Task automatically!
    print("\n" + "=" * 45)
    print("STATISTICAL ANALYSIS")
    print("=" * 45)

    tasks = df_results["Test_Task"].unique()

    for task in tasks:
        print(f"\n--- Analysis for Task: {task.upper()} ---")

        # Isolate the data for just this task
        task_df = df_results[df_results["Test_Task"] == task]
        conditions = task_df["Condition"].unique()

        grouped_data = {
            cond: task_df[task_df["Condition"] == cond]["Mean_Anxiety_Score"].values
            for cond in conditions
        }
        data_arrays = list(grouped_data.values())

        # We need at least 2 conditions and at least 3 samples per condition to run these tests reliably
        if len(conditions) > 1 and all(len(group) >= 3 for group in data_arrays):
            # Test for Normality (Shapiro-Wilk)
            normality_passed = True
            print("Normality check (Shapiro-Wilk):")
            for cond, data in grouped_data.items():
                stat, p = stats.shapiro(data)
                status = "pass" if p > 0.05 else "fail"
                if p <= 0.05:
                    normality_passed = False
                print(f"   - {cond.lower():<10}: p = {p:.4f} [{status}]")

            # Test for Equal Variances (Levene's Test)
            print("\nVariance check (Levene's):")
            stat, p_levene = stats.levene(*data_arrays)
            variance_passed = p_levene > 0.05
            status = "pass" if variance_passed else "fail"
            print(f"   - all groups: p = {p_levene:.4f} [{status}]\n")

            # Choose the appropriate test
            if normality_passed and variance_passed:
                print("Assumptions met. Running parametric One-Way ANOVA:")
                stat, p_value = stats.f_oneway(*data_arrays)
                test_name = "ANOVA"
            else:
                print("Assumptions violated. Running non-parametric Kruskal-Wallis:")
                stat, p_value = stats.kruskal(*data_arrays)
                test_name = "Kruskal-Wallis"

            # Final conclusion
            print(f"\n{test_name.lower()} Results:")
            print(f"Statistic: {stat:.3f}")
            print(f"p-value:   {p_value:.4f}")

            if p_value < 0.05:
                print(
                    f"Conclusion: Significant difference detected between auditory conditions for the {task.upper()} task."
                )
            else:
                print(
                    f"Conclusion: No significant difference detected for the {task.upper()} task."
                )

        else:
            print(
                f"Not enough conditions or samples per group to perform statistical testing for {task.upper()}."
            )
            print("(Need at least 2 conditions, and at least 3 runs per condition).")

    # Plotting the comparison
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    title = f"Average Predicted PSS Per Condition ({target_participant})"
    if target_test:
        title += f" - {target_test.lower()} task"
    fig.canvas.manager.set_window_title("EEG Condition Comparison")

    sns.boxplot(
        data=df_results,
        x="Condition",
        y="Mean_Anxiety_Score",
        hue="Test_Task",
        palette="magma",
        ax=ax,
        width=0.6,
    )

    sns.stripplot(
        data=df_results,
        x="Condition",
        y="Mean_Anxiety_Score",
        hue="Test_Task",
        color=".3",
        alpha=0.7,
        dodge=True,
        size=6,
        ax=ax,
    )

    # Clean up the legend
    handles, labels = ax.get_legend_handles_labels()
    half = len(handles) // 2
    if half > 0:
        ax.legend(handles[:half], labels[:half], title="Task")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted PSS Score (Saeed Regression)")
    ax.set_xlabel("Auditory Condition")

    plt.tight_layout()
    plt.show()

    return df_results


if __name__ == "__main__":
    DATA_DIR = "data/level_2"
    TARGET_PARTICIPANT = "Andy"

    TARGET_TEST = None

    if os.path.exists(DATA_DIR):
        df_summary = analyze_experiment_runs(DATA_DIR, TARGET_PARTICIPANT, TARGET_TEST)
    else:
        print(f"Error: directory '{DATA_DIR}' not found.")

