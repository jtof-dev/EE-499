import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def calculate_run_anxiety(file_path):
    """processes a single EEG file and returns the mean anxiety/stress scores."""
    df = pd.read_csv(file_path)

    df_freq = df.drop_duplicates(
        subset=["delta", "theta", "alphaLow", "betaLow", "gammaLow"]
    ).copy()
    df_clean = df_freq[df_freq["poorSignal"] == 0].copy()

    if df_clean.empty:
        return None

    # z-score normalization per band for the saeed composite
    for band in ["betaLow", "betaHigh", "gammaLow"]:
        std = df_clean[band].std()
        df_clean[f"{band}_z"] = (
            0
            if std == 0 or np.isnan(std)
            else (df_clean[band] - df_clean[band].mean()) / std
        )

    df_clean["saeed_stress_score"] = (
        df_clean["betaLow_z"] + df_clean["betaHigh_z"] + df_clean["gammaLow_z"]
    )

    total_beta = df_clean["betaLow"].mean() + df_clean["betaHigh"].mean()
    total_alpha = df_clean["alphaLow"].mean() + df_clean["alphaHigh"].mean()

    return {
        "mean_saeed_stress": df_clean["saeed_stress_score"].mean(),
        "mean_beta_alpha_ratio": total_beta / (total_alpha + 1e-5),
        "clean_epochs": len(df_clean),
    }


def analyze_experiment_runs(data_dir, target_participant, target_test=None):
    print(f"scanning '{data_dir}' for participant '{target_participant.lower()}'...")

    run_results = []

    for file in os.listdir(data_dir):
        if not file.endswith(".csv"):
            continue

        # parse based on: yyyymmdd_hhmm_participant_datatype_test_testcondition.csv
        filename_no_ext = file.replace(".csv", "")
        parts = filename_no_ext.split("_")

        # ensure it matches the expected 6-part structure
        if len(parts) != 6:
            continue

        date, time, participant, datatype, test_type, condition = parts

        # filter for the correct participant and ensure it's EEG data
        if participant != target_participant or datatype != "EEG":
            continue

        # optional: filter by specific test
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
        print("no valid data found to compare.")
        return None

    df_results = pd.DataFrame(run_results)

    # CLI summary table
    print("\naggregated anxiety levels")
    if target_test:
        print(f"filtered for task: {target_test.lower()}")

    summary_table = (
        df_results.groupby(["Test_Task", "Condition"])
        .agg(
            Runs=("Filename", "count"),
            Avg_Saeed_Score=("Mean_Anxiety_Score", "mean"),
        )
        .reset_index()
    )
    summary_table.columns = [c.lower() for c in summary_table.columns]
    print(summary_table.to_string(index=False, float_format="%.3f"))

    # statistical testing (assumption checks & auto-routing)
    print("\nstatistical analysis")

    # group the anxiety scores by condition
    conditions = df_results["Condition"].unique()
    grouped_data = {
        cond: df_results[df_results["Condition"] == cond]["Mean_Anxiety_Score"].values
        for cond in conditions
    }
    data_arrays = list(grouped_data.values())

    # we need at least 2 conditions and at least 3 samples per condition to run these tests reliably
    if len(conditions) > 1 and all(len(group) >= 3 for group in data_arrays):
        # test for normality (shapiro-wilk)
        # null hypothesis: data is drawn from a normal distribution.
        normality_passed = True
        print("\nnormality check (shapiro-wilk):")
        for cond, data in grouped_data.items():
            stat, p = stats.shapiro(data)
            status = "pass" if p > 0.05 else "fail"
            if p <= 0.05:
                normality_passed = False
            print(f"   - {cond.lower():<10}: p = {p:.4f} [{status}]")

        # test for equal variances (levene's test)
        # null hypothesis: all input samples are from populations with equal variances.
        print("\nvariance check (levene's):")
        stat, p_levene = stats.levene(*data_arrays)
        variance_passed = p_levene > 0.05
        status = "pass" if variance_passed else "fail"
        print(f"   - all groups: p = {p_levene:.4f} [{status}]\n")

        # choose the appropriate test
        if normality_passed and variance_passed:
            print("assumptions met. running parametric one-way ANOVA:")
            stat, p_value = stats.f_oneway(*data_arrays)
            test_name = "ANOVA"
        else:
            print("assumptions violated. running non-parametric kruskal-wallis:")
            stat, p_value = stats.kruskal(*data_arrays)
            test_name = "kruskal-wallis"

        # final conclusion
        print(f"\n{test_name.lower()} results:")
        print(f"statistic: {stat:.3f}")
        print(f"p-value:   {p_value:.4f}")

        if p_value < 0.05:
            print(
                "\nconclusion: significant difference detected between auditory conditions."
            )
        else:
            print(
                "\nconclusion: no significant difference detected (variance may be random)."
            )

    else:
        print(
            "not enough condition groups or samples per group to perform statistical testing."
        )
        print("(need at least 2 conditions, and at least 3 runs per condition).")

    # plotting the comparison
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    title = f"Average Anxiety Levels Per Condition ({target_participant})"
    if target_test:
        title += f" - {target_test.lower()} task"
    fig.canvas.manager.set_window_title("EEG condition comparison")

    sns.boxplot(
        data=df_results,
        x="Condition",
        y="Mean_Anxiety_Score",
        hue="Test_Task",
        palette="magma",
        ax=ax,
        width=0.6,
    )

    # dodge=true aligns the dots with the specific hue boxes
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

    # clean up the legend so we don't get double entries for the box/strip plots
    handles, labels = ax.get_legend_handles_labels()
    half = len(handles) // 2
    if half > 0:
        ax.legend(handles[:half], labels[:half], title="Task")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Saeed Stress Score (Z-Normalized)")
    ax.set_xlabel("Auditory Condition")

    plt.tight_layout()
    plt.show()

    return df_results


if __name__ == "__main__":
    DATA_DIR = "data/level_2"
    TARGET_PARTICIPANT = "Andy"

    # you can set this to "stroop", "reading", etc., or leave as none to analyze all tasks
    TARGET_TEST = None

    if os.path.exists(DATA_DIR):
        df_summary = analyze_experiment_runs(DATA_DIR, TARGET_PARTICIPANT, TARGET_TEST)
    else:
        print(f"error: directory '{DATA_DIR}' not found.")
