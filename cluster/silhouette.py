import numpy as np
from scipy.spatial.distance import cdist

#this score is a metrix used to measure how well a data point fits within it assigned 
#cluster compared to other clusters. it ranges from -1 to 1,
#where 1 data point is well clustered (far from other clusters)
#0 data point is on the border between two clusters
#-1 data point is misclassified, closer to another cluster than its own!

class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        we are initializing the class, no inputs required
        """
        pass #no parameters to score

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be a NumPy array")
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be a 2D array and y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows (samples).")
        
        """
        Compute pairwise distances between all points
        necessary to efficiently calculate the average intra-cluster distance (how close a point is to others in its cluster)
        and the average nearest-cluster distance (how far a point is from the closest other cluster)
        each row represents a data point, each column represents the distance to every other poin
        """
        distances = cdist(X, X)

        #unique cluster labels
        unique_labels = np.unique(y)

        #Initialize silhouette scores array 
        silhouette_scores = np.zeros(len(X))

        for i, point in enumerate(X):
            cluster = y[i] #get the cluster of the current point

            """
            compute the average intra-cluster distance
            y == cluster creates a Boolean mask that selects only the other points in the same cluster.
            np.arange(len(y) != i) ensures point i is excluded from its own cluster calculations.
            """ 
            in_cluster = (y == cluster) & (np.arange(len(y)) != i) #identifies all points belonging to the same cluster as point i, excluding itself.
            if np.sum(in_cluster) > 0:
                a_i = np.mean(distances[i, in_cluster])
            else:
                a_i = 0 #if its the only point in the cluster, set a(i) to 0. 
            
            """
            compute b(i), which is the average inter-cluster distance (nearest neighbor cluster)
            """
            b_i = np.inf #start with a large number

            for other_cluster in unique_labels:
                if other_cluster == cluster:
                    continue #skip own cluster

                out_cluster = (y == other_cluster)
                mean_distance = np.mean(distances[i, out_cluster])

                if mean_distance < b_i:
                    b_i = mean_distance #keep the smallest inter cluster distance
                
                silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
            
            return silhouette_scores

            



