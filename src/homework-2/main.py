import pandas as pd  # used to parse .csv's
from scipy import stats  # used for p-value calculations


def calculate_mean(data, use_harmonic=False):
    """
    calculates either the harmonic or arithmetic mean for a single dataset or a list of datasets
    defaults to arithmetic mean, but can be overwritten by passing `use_harmonic=True`
    returns either a mean or a list of means
    """
    # if the data is a pandas object, convert it to a list
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.tolist()

    # check if data is a list of lists, and not a pandas series or dataframe
    if isinstance(data, list) and data and isinstance(data[0], list):
        results = []
        for d in data:
            # if there are multiple datasets, recursively call calculate_mean() on each dataset
            results.append(
                calculate_mean(d, use_harmonic)
            )  # pass through `use_hamonic` to preserve the original state
        return results

    # from here, `data` is a single list
    n = len(data)
    if n == 0:
        return 0

    if use_harmonic:
        # calculate harmonic mean
        total_reciprocal = 0.0
        for x in data:
            if x == 0:
                return 0.0  # harmonic mean is 0 if any element is 0
            total_reciprocal += 1.0 / x

        return n / total_reciprocal
    else:
        # arithmetic mean
        if n == 0:
            return 0
        return sum(data) / n


def std_dev(data):
    """
    calculates population standard deviation from a list of data
    """
    # if the data is a pandas object, convert it to a list
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values.flatten().tolist()

    n = len(data)  # sanity check on data
    if n == 0:
        return 0

    mean = sum(data) / n
    # using equation from 'homework-2.pdf', split into 2 lines
    sq_diff = sum((x - mean) ** 2 for x in data)
    return (sq_diff / n) ** 0.5


def pooled_std_dev(data_pairs):
    """
    calculates the pooled standard deviation from a list of (sigma, n) pairings
    """
    k = len(data_pairs)  # sanity check on data
    if k < 2:
        return "error: need at least two pairings to pool"

    # split equation from 'homework-2.pdf' into numerator and denominator
    numerator = 0
    denominator = 0

    # do numerator and denominator calculations for each (sigma, n)
    for sigma, n in data_pairs:
        numerator += (n - 1) * (sigma**2)
        denominator += n - 1

    if denominator == 0:
        return 0

    return (numerator / denominator) ** 0.5  # take final square root


def t_test(data1, data2, use_harmonic=False):
    """
    performs t-test between two sets of data, and returns the t- and p-test from two inputted lists
    when calculating mu, there is an option to use harmonic mean over the arithmetic mean (defaults to arithmetic), but from what I understand there would never be a reason to do this
    """
    params = []

    # handle whether the inputs are datasets with needed parameters, or if they need to be calculated first
    for data in [data1, data2]:
        # if the data is a pandas object, convert it to a list
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.tolist()

        if isinstance(data, (list, tuple)) and not isinstance(data[0], (int, float)):
            params.append(data)
        else:
            mu = calculate_mean(data, use_harmonic)
            sigma = std_dev(data)
            n = len(data)
            params.append((mu, sigma, n))

    # from here, data1 and data2 should be formatted correctly in `params`
    (mu1, sigma1, n1), (mu2, sigma2, n2) = params

    # using the equation from 'homework-2.pdf', split up the t-test into three steps
    # 1. calculate the pooled standard deviation
    sigma_p = pooled_std_dev([(sigma1, n1), (sigma2, n2)])
    # 2. calculate standard error
    standard_error = sigma_p * ((1 / n1 + 1 / n2) ** 0.5)
    # 3. combine into final t-value
    t_value = (mu1 - mu2) / standard_error

    # use t-value to calculate p-value using scipy
    degees_freedom = n1 + n2 - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_value), degees_freedom))

    return t_value, p_value


def anova(datasets):
    """
    calculates the anova f-stat and p-value for 3 or more datasets
    """
    # if the data is a pandas object, convert it to a list
    if isinstance(datasets, (pd.DataFrame, pd.Series)):
        datasets = datasets.tolist()

    m = len(datasets)  # sanity check on datasets
    if m < 3:
        return "error: anova requires at least 3 datasets"

    # I followed the equations from 'homework-2.pdf' and did some more research online, so I believe that I set up the anova equations correctly
    # flatten datasets into one list
    all_observations = [item for sublist in datasets for item in sublist]
    N = len(all_observations)
    overall_mean = calculate_mean(all_observations)

    # calculate the sum of squares total
    sum_squares_total = sum((x - overall_mean) ** 2 for x in all_observations)

    # calculate the sum of squares between
    sum_squares_between = 0
    for data in datasets:
        n_j = len(data)
        group_mean = calculate_mean(data)
        sum_squares_between += n_j * (group_mean - overall_mean) ** 2

    # calculate the sume of squares within
    sum_squares_within = sum_squares_total - sum_squares_between

    degrees_freedom_between = m - 1
    degrees_freedom_within = N - m

    # calculate mean squares for the f-stat
    mean_squares_between = sum_squares_between / degrees_freedom_between
    mean_squares_within = sum_squares_within / degrees_freedom_within

    # calculate the f-stat and p-value
    f_stat = mean_squares_between / mean_squares_within
    p_value = 1 - stats.f.cdf(f_stat, degrees_freedom_between, degrees_freedom_within)

    return f_stat, p_value


def rmanova(datasets):
    """
    calculates the rmanova f-stat and p-value from an inputted dataset (list of lists)
    """
    # if the data is a pandas object, convert it to a list of lists
    if isinstance(datasets, pd.DataFrame):
        # handle any null values so that rmanova() is happy
        if datasets.isnull().values.any():
            datasets.fillna(0, inplace=True)
        datasets = datasets.values.tolist()

    num_rows = len(datasets)
    num_columns = len(datasets[0])

    all_values = [val for sub in datasets for val in sub]
    all_values_mean = calculate_mean(all_values)

    # I followed the equations from 'homework-2.pdf' and did some more research online, so I believe that I set up the rmanova equations correctly
    # calculate the sum of squares of subjects
    sum_squares_subjects = 0
    for data in datasets:
        data_mean = calculate_mean(data)
        sum_squares_subjects += num_columns * (data_mean - all_values_mean) ** 2

    # calculate the sum of squares of conditions
    sum_squares_conditions = 0
    for column in range(num_columns):
        column_data = [datasets[row][column] for row in range(num_rows)]
        column_mean = calculate_mean(column_data)
        sum_squares_conditions += num_rows * (column_mean - all_values_mean) ** 2

    # calculate the sum of squares total using the previously calculated values
    sum_squares_total = sum((x - all_values_mean) ** 2 for x in all_values)
    sum_squares_error = (
        sum_squares_total - sum_squares_conditions - sum_squares_subjects
    )

    degrees_freedom_conditions = num_columns - 1
    degrees_freedom_error = (num_columns - 1) * (num_rows - 1)

    # calculate mean squares for f-stat
    mean_squares_conditions = sum_squares_conditions / degrees_freedom_conditions
    mean_squares_error = sum_squares_error / degrees_freedom_error

    # calculate the f-stat and p-value
    f_stat = mean_squares_conditions / mean_squares_error
    p_value = 1 - stats.f.cdf(f_stat, degrees_freedom_conditions, degrees_freedom_error)

    return f_stat, p_value


def main():
    # reusable preprocessing for fitbit dataframes
    fitbit_participant_files = [
        f"sample-data/actigraph-and-fitbit/fitbit/{i}_FB_minuteSteps.csv"
        for i in range(1, 5)
    ]
    fb_dataframes = []  # since I will re-use the fitbit steps data, I will also store it in a dataframe for later

    for file_path in fitbit_participant_files:
        try:
            fb_steps_csv_daily = pd.read_csv(file_path)
            # manually set date formatting so pandas can read them correctly (or else it fails to detect the format)
            fb_steps_csv_daily["ActivityHour"] = pd.to_datetime(
                fb_steps_csv_daily["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p"
            )
            fb_steps_csv_daily["Date"] = fb_steps_csv_daily["ActivityHour"].dt.date
            fb_dataframes.append(fb_steps_csv_daily)  # store steps data for reuse
        except Exception as e:
            print(f"error reading {file_path}: {e}")

    # define the minutes columns, assuming they're the same for all files
    fb_steps_cols = (
        [col for col in fb_dataframes[0].columns if col.startswith("Steps")]
        if fb_dataframes
        else []
    )

    # 1. daily steps
    print("-------------\n daily steps\n-------------\n")

    daily_steps = []

    # iterate through all four daily steps .csv files and sum up number of steps
    for fb_steps_csv_daily in fb_dataframes:
        # group by day to calculate daily mean
        daily_groups = fb_steps_csv_daily.groupby("Date")

        for date, group in daily_groups:
            day_data = (
                group[fb_steps_cols].sum().sum()
            )  # add up the total steps for the day
            daily_steps.append(day_data)

    # take both means
    daily_steps_arith_mean = calculate_mean(daily_steps, use_harmonic=False)
    daily_steps_harmonic_mean = calculate_mean(daily_steps, use_harmonic=True)

    print(f"arith: {daily_steps_arith_mean}")
    print(f"harmonic: {daily_steps_harmonic_mean}")

    # -----------------
    # 2. group variance
    print("\n----------------\n group variance\n----------------\n")

    # variable to store (sigma, n) for each participant
    participant_stats = []

    # iterate through all four participants
    for steps_variance in fb_dataframes:  # reuse daily steps dataframes
        # calculate this participant's (sigma, n)
        sigma_i = std_dev(steps_variance[fb_steps_cols])
        n_i = len(steps_variance[fb_steps_cols].values.flatten().tolist())

        participant_stats.append([sigma_i, n_i])

    # calculate pooled_std_dev() on the collected pairs
    group_std_dev = pooled_std_dev(participant_stats)
    group_variance = group_std_dev**2  # variance is pooled standard deviation squared

    print(f"group pooled std dev: {group_std_dev}")
    print(f"group pooled variance: {group_variance}")

    # ------------------------
    # 3. comparing the devices
    print(
        "\n-----------------------\n comparing the devices\n-----------------------\n"
    )

    # variables to store fitbit and actigraph data
    fb_data_all = []
    ag_data_all = []

    # process fitbit steps data
    for fb_steps in fb_dataframes:  # reuse daily steps dataframes
        # combine the 60 steps columns into one steps list
        fb_steps_melt = fb_steps.melt(
            id_vars=["ActivityHour"],
            value_vars=fb_steps_cols,
            var_name="Minute",
            value_name="Steps",
        )

        fb_steps_melt["Minute"] = (
            fb_steps_melt["Minute"].str.replace("Steps", "").astype(int)
        )  # convert the 'Minute' column to integers

        fb_steps_melt["datetime"] = fb_steps_melt.apply(
            lambda row: row["ActivityHour"] + pd.Timedelta(minutes=row["Minute"]),
            axis=1,
        )  # turn the steps list dates into `datetime` by adding minutes

        # format data to be merged later
        fb_steps_final = fb_steps_melt[["datetime", "Steps"]].set_index("datetime")
        fb_data_all.append(fb_steps_final)

    # process actigraph steps data
    for i in range(1, 5):
        # actigraph data comes in two files per participant, so iterate through both
        for week in range(1, 3):
            ag_file = (
                f"sample-data/actigraph-and-fitbit/actigraph/{i}_AG_week{week}.csv"
            )

            with open(ag_file, "r") as f:
                header = [
                    next(f) for _ in range(10)
                ]  # remove 10 line header on the actigraph files

            # text match in the header to find start time and date
            ag_start_time_str = (
                [line for line in header if "Start Time" in line][0]
                .split(" ")[-1]
                .strip()
            )
            ag_start_date_str = (
                [line for line in header if "Start Date" in line][0]
                .split(" ")[-1]
                .strip()
            )

            ag_start_datetime = pd.to_datetime(
                f"{ag_start_date_str} {ag_start_time_str}"
            )  # combine the time and day into a datetime

            # use pandas to read the steps data
            ag_steps = pd.read_csv(
                ag_file,
                skiprows=10,
                header=None,
                usecols=[3],
                names=["Steps"],
            )

            # format data to be merged later
            ag_steps.index = pd.date_range(
                start=ag_start_datetime, periods=len(ag_steps), freq="min"
            )
            ag_data_all.append(ag_steps)

    # use pandas to merge the two datasets into one variable with suffixes differentiating them
    fb_data_merged = pd.concat(fb_data_all)
    ag_data_merged = pd.concat(ag_data_all)
    merged_data = pd.merge(
        fb_data_merged,
        ag_data_merged,
        left_index=True,
        right_index=True,
        suffixes=("_fitbit", "_actigraph"),
    )

    # now run the t-test on the processed data
    t_stat_p3, p_value_p3 = t_test(
        merged_data["Steps_fitbit"], merged_data["Steps_actigraph"]
    )

    print(f"t-stat: {t_stat_p3}")
    print(f"p-value: {p_value_p3}")

    # -------------------
    # 4. weekend warriors
    print("\n------------------\n weekend warriors\n------------------\n")

    fb_steps = pd.concat(
        fb_dataframes, ignore_index=True
    )  # I know that this is a re-used variable

    # seperate the minute steps columns
    fb_steps["hourly_steps"] = fb_steps[fb_steps_cols].sum(
        axis=1
    )  # combine into hourly steps

    fb_steps_formatted = (
        fb_steps.groupby("Date")["hourly_steps"].sum().reset_index()
    )  # sort by date
    fb_steps_formatted.rename(
        columns={"hourly_steps": "total_daily_steps"}, inplace=True
    )  # reformat hourly --> daily

    fb_steps_formatted["day_of_week"] = pd.to_datetime(
        fb_steps_formatted["Date"]
    ).dt.weekday  # add a days column

    # format for anova()
    fb_steps_anova = fb_steps_formatted.groupby("day_of_week")[
        "total_daily_steps"
    ].apply(list)

    # anova() already handles errors, so just directly pass in the data
    f_stat_p4, p_value_p4 = anova(fb_steps_anova)
    print(f"f-stat: {f_stat_p4}")
    print(f"p-value: {p_value_p4}")

    # --------------
    # 5. seasonality
    print("\n-------------\n seasonality\n-------------\n")

    # now we are working with the multiyear/ data
    # do similar .csv processing as above
    multi_steps_file = "sample-data/multiyear/dailySteps.csv"
    multi_steps = pd.read_csv(multi_steps_file)
    multi_steps["ActivityDay"] = pd.to_datetime(
        multi_steps["ActivityDay"]
    )  # convert into dataframes

    # process by year and month
    multi_steps["Year"] = multi_steps["ActivityDay"].dt.year
    multi_steps["Month"] = multi_steps["ActivityDay"].dt.month

    # group by month and year and calculate mean
    multi_monthly_avg_steps = (
        multi_steps.groupby(["Year", "Month"])["StepTotal"]
        .agg(
            calculate_mean
        )  # this line runs calculate_mean() on each month (arithmetic)
        .reset_index()
    )

    multi_steps_pivot = multi_monthly_avg_steps.pivot(
        index="Year", columns="Month", values="StepTotal"
    )  # adjusts the dataframe to sort by year and month, with monthly averaged steps

    # now run it through rmanova()
    f_stat_p5, p_value_p5 = rmanova(multi_steps_pivot)
    print(f"f-stat: {f_stat_p5}")
    print(f"p-value: {p_value_p5}")


if __name__ == "__main__":
    main()

