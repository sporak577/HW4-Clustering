# Write your k-means unit tests here
import pytest 
import numpy as np
from cluster.kmeans import KMeans

#Test 1: kmeans initialization
def test_kmeans_init():
    kmeans = KMeans(k=3)
    assert kmeans.k == 3
    assert kmeans.centroids is None
    assert kmeans.labels is None

#Test 2: running kmeans on synthetic data
def test_kmeans_fit():
    X = np.array([
        [1,2], [1.5, 1.8], [5, 8],
        [8, 8], [1, 0.6], [9, 11]
    ])

    kmeans = KMeans(k=2)
    kmeans.fit(X)

    #Check if centroids are not None
    assert kmeans.centroids is not None
    assert len(kmeans.centroids) == 2 # must have 2 clusters

    #ensure all points are assigned in a cluster
    assert len(kmeans.labels) == len(X)

#Test 3: predict method
def test_kmeans_predict():
    X_train = np.array([
        [1,2], [1.5, 1.8], [5, 8],
        [8, 8], [1, 0.6], [9, 11]
    ])
    X_test = np.array([
        [2, 3], [6, 9]
    ])

    kmeans = KMeans(k=2)
    kmeans.fit(X_train)
    predictions = kmeans.predict(X_test)

    #ensure predictions return a label for each test point
    assert len(predictions) == len(X_test)
    assert predictions.ndim == 1 #1D array of labels

def test_kmeans_error():
    X = np.array([
        [1,2], [1.5, 1.8], [5, 8],
        [8, 8], [1, 0.6], [9, 11]
    ])
    kmeans = KMeans(k=2)
    kmeans.fit(X)

    error = kmeans.get_error()
    assert isinstance(error, float)
    assert error >= 0 #error should never be negative