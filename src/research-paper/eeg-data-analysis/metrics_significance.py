import os
import re

import pandas as pd
import scipy.stats as stats

# configuration
DATA_DIR = "data/level_2"
CONDITIONS = ["Silent", "WhiteNoise", "Music", "MusicNL"]
DISCARD_SECONDS = 60
TARGET_PARTICIPANT_REGEX = r"^Andy$"
TARGET_TEST_TYPE = "Stroop"  # change to "Stroop" or "Typing"

# preparation
run_stats = []
regex_pattern = re.compile(TARGET_PARTICIPANT_REGEX)

print(
    f"loading data and running final statistical analysis for {TARGET_PARTICIPANT_REGEX.strip('^$').lower()} on {TARGET_TEST_TYPE.lower()} test...\n"
)

# iterate through the directory to extract individual run statistics
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
        or test_type != TARGET_TEST_TYPE
        or condition not in CONDITIONS
    ):
        continue

    file_path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(file_path)

    # calculate elapsed seconds and discard settling time
    df["seconds_elapsed"] = (
        ((df["timestampMs"] - df["timestampMs"].iloc[0]) / 1000).round().astype(int)
    )
    df = df[df["seconds_elapsed"] >= DISCARD_SECONDS].copy()

    if df.empty:
        continue

    # test-specific calculations
    if TARGET_TEST_TYPE == "Stroop":
        df["throughput"] = df["keys_pressed"].rolling(window=10, min_periods=1).mean()
        total_actions = df["keys_pressed"].sum()
        total_errors = df["errors"].sum()

        if total_actions > 0:
            correct = max(0, total_actions - total_errors)
            accuracy = min(100.0, (correct / total_actions) * 100)
        else:
            accuracy = 0

    elif TARGET_TEST_TYPE == "Typing":
        df["errors"] = df["errors"].clip(upper=1)
        df["words_completed"] = df["total_words"].diff().fillna(0)
        df["throughput"] = (
            df["words_completed"].rolling(window=10, min_periods=1).mean()
        )

        total_correct = df["words_completed"].sum()
        total_errors = df["errors"].sum()
        total_actions = total_correct + total_errors

        if total_actions > 0:
            accuracy = (total_correct / total_actions) * 100
        else:
            accuracy = 0

    run_stats.append(
        {
            "Condition": condition,
            "Accuracy": accuracy,
            "Throughput": df["throughput"].mean(),
        }
    )

df_stats = pd.DataFrame(run_stats)

if df_stats.empty:
    print("no data found. check your directory or filters.")
    exit()

print("\nshapiro-wilk test (are the distributions normal?)")
print("null hypothesis: the data is normally distributed (p > 0.05)\n")

print("\nlevene's test (is the variance roughly equal across groups?)")
print("null hypothesis: all groups have equal variance (p > 0.05)\n")

print("\nnext steps:")
print("if all conditions are normal and have equal variances -> run an ANOVA.")
print(
    "if any condition is not normal or variance is unequal -> run a kruskal-wallis test.\n"
)
print("-" * 50)

# run automated checks and statistics for each metric
for metric in ["Accuracy", "Throughput"]:
    print(f"\n>>> analyzing {metric.lower()} <<<")

    # group data
    groups = {
        cond: df_stats[df_stats["Condition"] == cond][metric].values
        for cond in CONDITIONS
        if len(df_stats[df_stats["Condition"] == cond]) > 0
    }

    data_arrays = list(groups.values())
    active_conditions = list(groups.keys())

    if len(data_arrays) < 2:
        print("not enough data to compare groups.")
        continue

    # check shapiro-wilk assumptions
    is_normal = True
    for cond, cond_data in groups.items():
        if len(cond_data) >= 3:
            stat, p_val = stats.shapiro(cond_data)
            normal_status = "normal" if p_val > 0.05 else "not normal"
            print(f"shapiro ({cond.lower()}): {normal_status} (p = {p_val:.4f})")
            if p_val < 0.05:
                is_normal = False
        else:
            print(f"shapiro ({cond.lower()}): not enough data (n={len(cond_data)})")
            is_normal = False

    # check levene assumptions
    try:
        stat, p_levene = stats.levene(*data_arrays)
        is_equal_var = p_levene >= 0.05
        var_status = "equal variances" if is_equal_var else "unequal variances"
        print(f"levene (all groups): {var_status} (p = {p_levene:.4f})")
    except ValueError:
        is_equal_var = False
        print("levene: failed to calculate variance.")

    # route to the correct statistical test based on assumptions
    if is_normal and is_equal_var:
        print(f"\n{metric.lower()} (method: one-way ANOVA)")
        f_stat, p_val = stats.f_oneway(*data_arrays)

        if p_val < 0.05:
            print(f"significant difference found (p = {p_val:.4f})")
            print("audio condition had a measurable impact. running post-hoc test...\n")

            try:
                tukey_result = stats.tukey_hsd(*data_arrays)
                print("\ntukey HSD post-hoc results")
                print(f"condition order: {[c.lower() for c in active_conditions]}")
                print(tukey_result)
            except AttributeError:
                print(
                    "note: update scipy to >=1.8.0 to view tukey HSD results automatically."
                )
        else:
            print(f"no significant difference (p = {p_val:.4f})")
            print(
                "any variations between audio conditions are likely just random chance."
            )

    else:
        print(f"\n{metric.lower()} (method: kruskal-wallis h test)")
        h_stat, p_val = stats.kruskal(*data_arrays)

        if p_val < 0.05:
            print(f"significant difference found (p = {p_val:.4f})")
            print("audio condition had a measurable impact.")
            print(
                "(note: for strict non-parametric pairwise comparisons, dunn's test via the scikit-posthocs library is recommended)."
            )
        else:
            print(f"no significant difference (p = {p_val:.4f})")
            print(
                "any variations between audio conditions are likely just random chance."
            )

    print("-" * 50)
