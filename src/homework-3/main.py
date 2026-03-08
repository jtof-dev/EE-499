import random
import math


def kmeans_single(X, k):
    centroids = random.sample(X, k)
    clusters = [[] for _ in range(k)]

    for row_idx, row in enumerate(X):
        distances = []
        for c in centroids:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(row, c)))
            distances.append(dist)

        cluster_idx = distances.index(min(distances))
        clusters[cluster_idx].append(row)

    new_centroids = []
    for cluster_points in clusters:
        if not cluster_points:
            new_centroids.append([0] * len(X[0]))
            continue

    new_centroids = []
    for cluster_points in clusters:
        if not cluster_points:
            new_centroids.append([0] * len(X[0]))
            continue

        dim_count = len(cluster_points[0])
        centroid = [
            sum(p[i] for p in cluster_points) / len(cluster_points)
            for i in range(dim_count)
        ]
        new_centroids.append(centroid)

    return new_centroids, clusters


def main():
    print("Hello from homework-3!")


if __name__ == "__main__":
    main()
