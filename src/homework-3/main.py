# I ran this script using `uv run main.py`, and to install the two dependencies run `uv sync`
# if you don't want to use `uv`, the only packages used are `pandas` and `scikit-learn`
import pandas as pd
import random
import math

# scikit-learn is only used to compare results against my hand-written kmeans() and knn() functions
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.neighbors import KNeighborsClassifier


def euclidean_distance(point1, point2):
    """
    calculates the euclidean distance between two data points
    input: point1 => list or tuple of coordinates; point2 => list or tuple of coordinates
    output: float distance
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def kmeans(X, k):
    """
    performs a single-pass k-means clustering analysis
    input: X => list or dataframe of data; k => int of number of clusters
    output: a tuple, (clusters, centroids)
    """

    # convert dataframe to a list of lists if necessary
    data = X.values.tolist() if hasattr(X, "values") else X

    # initialize centroids at random
    centroids = random.sample(data, k)
    clusters = [[] for _ in range(k)]

    # single pass assignment
    for point in data:
        # calculate euclidean distances
        distances = [euclidean_distance(point, c) for c in centroids]
        # do the assignments
        closest_index = distances.index(min(distances))
        clusters[closest_index].append(point)

    # calculate final centroids
    final_centroids = []
    for cluster in clusters:
        if not cluster:  # error handling
            final_centroids.append(None)
            continue
        # calculate each centroid
        num_dims = len(cluster[0])
        centroid = [sum(p[i] for p in cluster) / len(cluster) for i in range(num_dims)]
        final_centroids.append(centroid)

    return clusters, final_centroids


def knn(training_set, labels, new_point, k):
    """
    classifies a new data point based on the k-nearest neighbors in a training set
    input: training_set => list or dataframe of data; labels => list or series of classifications for the training set; new_point => list or series of an unclassified record; k => int (odd number) representing the number of neighbors
    output: the predicted class for the new point
    """

    # convert inputs to standard lists if they are pandas objects
    train_data = (
        training_set.values.tolist()
        if hasattr(training_set, "values")
        else training_set
    )
    train_labels = labels.tolist() if hasattr(labels, "tolist") else labels
    point = new_point.tolist() if hasattr(new_point, "tolist") else new_point

    distances = []
    for i in range(len(train_data)):
        # calculate euclidean distance
        dist = euclidean_distance(train_data[i], point)
        distances.append((dist, train_labels[i]))

    # sort by distance and take top k
    distances.sort(key=lambda x: x[0])
    neighbors_labels = [distances[i][1] for i in range(k)]

    # return most frequent label
    return max(set(neighbors_labels), key=neighbors_labels.count)


def cpa(time_series):
    """
    identifies the first dominant change point in a single-variable time series
    input: time_series => list or series of observations with equal time steps
    output: int index of the first dominant change point
    """

    # convert pandas series to list
    data = time_series.tolist() if hasattr(time_series, "tolist") else time_series

    # store total observations
    n = len(data)
    max_diff = -1
    change_point_index = 0

    # iterate through every split point in the time series
    for i in range(1, n):
        # split into left and right halves
        part1 = data[:i]
        part2 = data[i:]

        # calculate means in each segment
        mean1 = sum(part1) / len(part1)
        mean2 = sum(part2) / len(part2)

        diff = abs(mean1 - mean2)

        # keep track of the largest diff, and eventually return the index of that diff
        if diff > max_diff:
            max_diff = diff
            change_point_index = i

    return change_point_index


def main():
    # load the data
    df = pd.read_csv("dailySteps.csv")

    # remove zero step days
    df = df[df["StepTotal"] > 0].reset_index(drop=True)
    steps_list = df["StepTotal"].tolist()
    dates_list = df["ActivityDay"].tolist()

    # ------------------
    # 1. k-means and knn
    print("-----------------\n k-means and knn\n-----------------\n")

    # define variables for k-means
    X = df[["StepTotal"]].values.tolist()
    k = 3  # grouping into 3 categories

    # custom k-means
    custom_clusters, custom_centroids = kmeans(df[["StepTotal"]], k)

    # create a labels list from the custom clusters
    data_list_for_labels = df[["StepTotal"]].values.tolist()
    custom_labels = [0] * len(data_list_for_labels)
    for i, point in enumerate(data_list_for_labels):
        for cluster_idx, cluster in enumerate(custom_clusters):
            if point in cluster:
                custom_labels[i] = cluster_idx
                break

    # flatten custom centroids for easy reading
    c_centers = sorted([round(c[0]) for c in custom_centroids])

    # scikit-learn k-means
    sk_kmeans = SklearnKMeans(n_clusters=k, n_init="auto", random_state=42)
    sk_kmeans.fit(X)
    # extract and sort centers
    sk_centers = sorted([round(c[0]) for c in sk_kmeans.cluster_centers_])

    print(f" k-means:")
    print(f"  custom centers: {c_centers}")
    print(f"  sklearn centers: {sk_centers}")

    # define variables for knn
    n_neighbors = 5

    # hypothetical test points for different activity levels
    test_points = [[2000], [8000], [14000]]

    print(f"\n k-nearest neighbors:")
    print(f"  testing both models with new data points (k={n_neighbors}):")

    # scikit-learn knn can predict all points at once
    sk_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    sk_knn.fit(X, sk_kmeans.labels_)
    sk_predictions = sk_knn.predict(test_points)

    # loop through test points for custom knn and comparison
    for i, point in enumerate(test_points):
        # calculate custom knn for each point
        custom_prediction = knn(
            df[["StepTotal"]], custom_labels, point, k=n_neighbors
        )

        # get the corresponding scikit-learn prediction
        sk_prediction = sk_predictions[i]

        print(f"\n - for point: {point[0]} steps")
        print(f"    custom prediction:  cluster {custom_prediction}")
        print(f"    sklearn prediction: cluster {sk_prediction}")

    # ------------------------
    # 2. change point analysis
    print(
        "\n-----------------------\n change point analysis\n-----------------------\n"
    )

    change_points = []
    # store the start and end indices of the segments we need to check
    segments_to_check = [(0, len(steps_list))]

    while len(change_points) < 8 and segments_to_check:
        start, end = segments_to_check.pop(0)
        segment = steps_list[start:end]

        if len(segment) < 2:
            continue

        local_cp = cpa(segment)

        if local_cp > 0:
            global_cp = start + local_cp
            change_points.append(global_cp)

            # store the left and right splits for further analysis
            segments_to_check.append((start, global_cp))
            segments_to_check.append((global_cp, end))

    # sort chronologically
    change_points.sort()

    print(f"identified {len(change_points)} change points:")
    for i, cp_idx in enumerate(change_points):
        date = dates_list[cp_idx]
        steps = steps_list[cp_idx]
        print(
            f"  change point {i + 1}: index {cp_idx:3d} | date: {date} | steps shifted at: {steps}"
        )


if __name__ == "__main__":
    main()
