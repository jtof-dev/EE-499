import pandas as pd
from scipy import stats

dailyCalories = pd.read_csv("sample-data/multiyear/dailyCalories.csv")
dailyCaloriesShort = dailyCalories["Calories"].tolist()


def harmonic_mean(datasets):
    """
    calculates the harmonic mean across N datasets
    takes data input of a list of lists
    """
    results = []

    # parsing the inputted datasets
    for data in datasets:
        # if there is no data, return 0
        n = len(data)
        if n == 0:
            results.append(0)
            continue

        # calculating the harmonic mean
        try:
            # using equation from homework-2.pdf
            h_mean = calculate_mean(data, use_harmonic=True)
            results.append(h_mean)
        except ZeroDivisionError:
            results.append("error: divided by zero")


def std_dev(data):
    """
    calculates population standard deviation
    takes data input of a list
    """
    n = len(data)
    if n == 0:
        return 0

    mean = sum(data) / n
    # using equation from homework-2.pdf
    sq_diff = sum((x - mean) ** 2 for x in data)
    return (sq_diff / n) ** 0.5


def pooled_std_dev(data_pairs):
    """
    calculates the pooled standard deviation from (sigma, n) pairings.
    Input: A list of lists, [(sigma1, n1), (sigma2, n2)]
    """
    # parsing the inputted data
    k = len(data_pairs)
    if k < 2:
        return "error: need at least two pairings to pool"

    # using equation from homework-2.pdf
    numerator = 0
    denominator = 0

    for sigma, n in data_pairs:
        numerator += (n - 1) * (sigma**2)
        denominator += n - 1

    if denominator == 0:
        return 0

    return (numerator / denominator) ** 0.5


def calculate_mean(data, use_harmonic=False):
    """
    calculates either the harmonic or arithmetic mean
    defaults to arithmetic mean, since that is used more frequently within this script
    """

    if use_harmonic:
        n = len(data)
        # harmonic mean
        return n / sum(1.0 / x for x in data if x != 0)
    else:
        # arithmetic mean
        return sum(data) / len(data)


def t_test(data1, data2):
    """
    performs t-test between two sets of data, and returns the t-test and p-test
    takes two lists and a true/false variable
    """
    params = []

    # handle whether the inputs are datasets with needed parameters, or if they need to be calculated first
    for d in [data1, data2]:
        if isinstance(d, (list, tuple)) and not isinstance(d[0], (int, float)):
            params.append(d)
        else:
            mu = calculate_mean(d)
            sigma = std_dev(d)
            n = len(d)
            params.append((mu, sigma, n))

    (mu1, sigma1, n1), (mu2, sigma2, n2) = params

    # first, calculate the pooled standard deviation
    sigma_p = pooled_std_dev([(sigma1, n1), (sigma2, n2)])

    # second, calculate t-value using equation from homework-2.pdf
    standard_error = sigma_p * ((1 / n1 + 1 / n2) ** 0.5)
    t_value = (mu1 - mu2) / standard_error

    # third, take t-value and calculate p-value using scipy
    degees_freedom = n1 + n2 - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_value), degees_freedom))

    return t_value, p_value


def anova(datasets):
    """
    calculates the anova f-stat and p-value for 3 or more datasets
    """
    # sanity check on inputted datasets
    m = len(datasets)
    if m < 3:
        return "error: anova requires at least 3 datasets"

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

    f_stat = mean_squares_between / mean_squares_within

    p_value = 1 - stats.f.cdf(f_stat, degrees_freedom_between, degrees_freedom_within)

    return f_stat, p_value


def rmanova(datasets):
    """
    calculates the rmanova f-stat and p-value for a given dataset
    input: a list of lists
    """
    num_rows = len(datasets)
    num_columns = len(datasets[0])

    all_values = [val for sub in datasets for val in sub]
    all_values_mean = calculate_mean(all_values)

    sum_squares_subjects = 0
    for data in datasets:
        data_mean = calculate_mean(data)
        sum_squares_subjects += num_columns * (data_mean - all_values_mean) ** 2

    sum_squares_conditions = 0
    for column in range(num_columns):
        column_data = [datasets[row][column] for row in range(num_rows)]
        column_mean = calculate_mean(column_data)
        sum_squares_conditions += num_rows * (column_mean - all_values_mean) ** 2

    sum_squares_total = sum((x - all_values_mean) ** 2 for x in all_values)
    sum_squares_error = (
        sum_squares_total - sum_squares_conditions - sum_squares_subjects
    )

    degrees_freedom_conditions = num_columns - 1
    degrees_freedom_error = (num_columns - 1) * (num_rows - 1)

    mean_squares_conditions = sum_squares_conditions / degrees_freedom_conditions
    mean_squares_error = sum_squares_error / degrees_freedom_error

    f_stat = mean_squares_conditions / mean_squares_error
    p_value = 1 - stats.f.cdf(f_stat, degrees_freedom_conditions, degrees_freedom_error)

    return f_stat, p_value


def main():
    # 1. daily steps
    fb_steps_csv = pd.read_csv(
        "sample-data/actigraph-and-fitbit/fitbit/1_FB_minuteSteps.csv"
    )

    # fb_steps_cols = [col for col in fb_steps_csv.columns if col.startswith("Steps")]
    # fb_steps_csv["ActivityHour"] = pd.to_datetime(fb_steps_csv["ActivityHour"])
    fb_steps_csv["ActivityHour"] = pd.to_datetime(
        fb_steps_csv["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p"
    )
    fb_steps_csv["Date"] = fb_steps_csv["ActivityHour"].dt.date

    steps_cols = fb_steps_csv.select_dtypes(include=["number"]).columns.tolist()
    daily_totals = fb_steps_csv.groupby("Date")[steps_cols].sum()

    results = []

    for col in daily_totals.columns:
        steps = daily_totals[col].tolist()

        steps_mean_arithmetic = calculate_mean(steps, use_harmonic=False)
        steps_mean_harmonic = calculate_mean(steps, use_harmonic=True)

        # print(f"{col:<15} | {steps_mean_arithmetic:<18.2f} | {steps_mean_harmonic:<15.2f}")

        results.append(
            {
                "Participant": col,
                "Arithmetic Mean": steps_mean_arithmetic,
                "Harmonic Mean": steps_mean_harmonic,
            }
        )

    results_fb_steps = pd.DataFrame(results)
    print(results_fb_steps.head())


if __name__ == "__main__":
    main()
