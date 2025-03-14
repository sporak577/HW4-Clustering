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
    