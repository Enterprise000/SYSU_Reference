from sklearn.datasets import load_iris
import numpy as np
from collections import deque
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def kmeans(X, y, k):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for i in range(300):
        # 每次迭代计算距离，根据距离确定标签
        distances = np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        centroids = []
        for i in range(k):
            if np.any(labels == i):
                centroids.append(X[labels == i].mean(axis=0))
            else:
                centroids.append(X[np.random.choice(n_samples)])
    return centroids, labels


def Near(X, eps, i):
    return np.where(np.linalg.norm(X - X[i], axis=1) <= eps)[0]


def cluster(X, eps, sample, i, near, id, labels):
    labels[i] = id
    queue = deque(near)
    while queue:
        point = queue.popleft()
        if labels[point] != -1:
            continue
        labels[point] = id
        near_new = Near(X, eps, point)
        queue.extend(set(near_new) - set(queue))


def DBSCAN(X, y, eps, sample):
    sample_num = X.shape[0]
    labels = -1 * np.ones(sample_num)
    id = 0

    for i in range(sample_num):
        if labels[i] != -1:
            continue
        near = Near(X, eps, i)
        if len(near) < sample:
            labels[i] = -1
        else:
            cluster(X, eps, sample, i, near, id, labels)
            id += 1

    return labels


def kmeans_evaluate(X, y, k):
    centroids, labels = kmeans(X, y, k)
    accuracy = accuracy_score(y, labels)
    ari = adjusted_rand_score(y, labels)
    silhouette = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    return accuracy, silhouette, ch_score,ari


def DBSCAN_evaluate(X, y, eps, sample):
    labels = DBSCAN(X, y, eps, sample)
    labels_valid = labels[labels != -1]
    X_valid = X[labels != -1]
    accuracy = accuracy_score(y, labels)
    ari = adjusted_rand_score(y, labels)
    silhouette = silhouette_score(X_valid, labels_valid)
    ch_score = calinski_harabasz_score(X_valid, labels_valid)
    return accuracy, silhouette, ch_score,ari


iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target


k_range = [2, 3, 5]
for k in k_range:
    accuracy, silhouette, ch_score,ari = kmeans_evaluate(X, y, k)
    print(f"K-means (k={k}) - Accuracy: {accuracy:.3f}, Silhouette: {silhouette:.3f}, Calinski-Harabasz: {ch_score:.3f}, ARI:{ari}")

eps_range = [0.3, 0.5, 0.7]
sample_range = [3, 5, 7]
for eps in eps_range:
    for sample in sample_range:
        accuracy, silhouette, ch_score, ari = DBSCAN_evaluate(X, y, eps, sample)
        print(f"DBSCAN (eps={eps},sample = {sample}) - Accuracy: {accuracy:.3f}, Silhouette: {silhouette:.3f}, Calinski-Harabasz: {ch_score:.3f},ARI:{ari}")