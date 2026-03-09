import random
import math


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
        distances = [
            math.sqrt(sum((a - b) ** 2 for a, b in zip(point, c))) for c in centroids
        ]
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
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(train_data[i], point)))
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

    n = len(data)
    max_diff = -1
    change_point_index = 0

    for i in range(1, n):
        part1 = data[:i]
        part2 = data[i:]

        mean1 = sum(part1) / len(part1)
        mean2 = sum(part2) / len(part2)

        diff = abs(mean1 - mean2)

        if diff > max_diff:
            max_diff = diff
            change_point_index = i

    return change_point_index


def main():
    print("Hello from homework-3!")


if __name__ == "__main__":
    main()
