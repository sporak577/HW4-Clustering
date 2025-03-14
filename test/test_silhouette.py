# write your silhouette score unit tests here

import pytest
import numpy as np
from cluster.silhouette import Silhouette
from sklearn.metrics import silhouette_score #import sklearn for comparison

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cluster.silhouette import Silhouette

#test 1: silhouette initialization
def test_silhouette_init():
    silhouette = Silhouette()
    assert isinstance(silhouette, Silhouette)

#test 2:computing silhouette score on valid data
    X = np.array([
         [1, 2], [1.5, 1.8], [5, 8],
        [8, 8], [1, 0.6], [9, 11]
    ])
    y = np.array([1, 1, 0, 0, 1, 0]) #cluster assignments

    silhouette = Silhouette()
    scores = silhouette_score(X, y)

    silhouette = Silhouette()
    scores = silhouette.score(X, y)

    assert len(scores) == len(X)
    assert all(-1 <= s <= 1 for s in scores) #scores must be in range [-1, 1]

#test 3: compare with sklearn's silhouette score
def test_silhouette_vs_sklearn():
    X = np.array([
        [1, 2], [1.5, 1.8], [5, 8],
        [8, 8], [1, 0.6], [9, 11]
    ])
    y = np.array([1, 1, 0, 0, 1, 0])

    silhouette= Silhouette()
    my_scores = silhouette.score(X, y)
    sklearn_scores = silhouette_score(X, y)

    assert np.isclose(np.mean(my_scores),sklearn_scores, atol=0.1) #allow small error margin

#here tesing an edge case, a single cluster
def test_silhouette_single_cluster():
    X = np.array([
        [1, 2], [1.5, 1.8], [1, 0.6]
    ])
    y = np.array([0, 0, 0]) #all points in one cluster

    silhouette = Silhouette()
    scores = silhouette.score(X, y)

    #Silhouette score should be 0 if all points are in the same cluster
    assert all(s == 0 for s in scores)
