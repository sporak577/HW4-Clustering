import numpy as np
from scipy.spatial.distance import cdist
'''
as always, I am using ChatGPT to help me understand the code and complete it. 

some comments that helped me understand the code:

tol defines how close to the exact solution the algorithm needs to get before stopping, so acts as a convergence threshold, 
allowing early stopping when further iterations bring only marginal improvements. without tol the algorithm would continue 
until reaching max_iter, even if centroids are already stable. 
due to float point precision errors direct comparisons between floats can be unreliable
'''

class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int) or k <= 0:
            #checks if k is an instance of the int class
            raise ValueError("k must be a positive integer.")
        
        #int are a subset of floats, meaning an int can be used anywhere a float is expected. 
        #we would still want an integer value e.g. tol = 1 instead of tol = 1.0 to be accepted as a valid input. 
        if not isinstance(tol, (float, int)) or tol < 0:
            #checks value for the variable tol, if it is a float or integer
            raise ValueError()
        
        #max_iter should be an integer, as this is the number of iteration before the algorithm stops. 
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        
        #here I initialize the instance attributes of the Kmeans class, making them available for later use
        #the methods like fit(), predict() etc. access these attributes without needing to pass them as arguments every time.  
        #if we wouldn't store these attributes in self, we would have to pass them around manually. for example self.centroids()
        #every time we call predict(), we would have to re-run fit() which is simply inefficient
        self.k = k 
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        
    


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        #the input data set has to be 2D, because the rows are the samples and the columns are the features we are interested in. 
        #kmeans needs multiple features to measure distances between points and form clusters. 
        if not isinstance(mat, np.ndarray) or mat.ndim!= 2:
            raise ValueError("Input data must be a 2-D numpy array.")
        
        n_samples, n_features = mat.shape
        
        #randomly initialize centroids from the dataset
        np.random.seed(42) #for reproducibility
        #selects k unique random indices from the dataset, replace=False ensures we don't pick the same data point twice. 
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = mat[random_indices]

        for _ in range(self.max_iter): #just repeats for the amount of iterables. no need for index.
            #compute distances and assign cluters
            distances = cdist(mat, self.centroids) #takes euclidean distance between each data point and each centroid

            #labels is a 1D array where each value is the cluster assignment for a sample, based on np.argim which finds
            #the index of the closest centroid for each data point. 
            labels = np.argmin(distances, axis=1)

            #compute new centroids, as the mean of all assigned points in each cluster. 
            #labels is a 1D array where each value represents the assigned cluster for each data point. 
            #labels == i creates a boolean mask that selects only the points assigned to cluster i. 
            new_centroids = np.array([mat[labels == i].mean(axis=0) for i in range(self.k)])

            #check for convergece, how close new centroid assigned is to the previous one
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break #this immediately stops the loop and skips any remaining iterations. 
            
            #update self.centroid if needed
            self.centroids = new_centroids
        
        self.labels = labels #store labels for error calculation, as labels link each data point to its assigned cluster

       

    #this method does not modify centroids, only classifies new points. 
    #this method classifies new data based on the trained centroids without modifying them. 
    #this is used for assigning new data points to clusters. 
    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if not isinstance(mat, np.ndarray) or mat.ncim != 2:
            raise ValueError("Input data must be a 2D Numpy array.")
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. call 'fit first.")
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("Feature mismatch: input data must have the same number of features as training data. ")
    
        distances = cdist(mat, self.centroids)
        return np.argmin(distances, axis = 1)


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.labels is None:
            raise ValueError
        
        #compute the squared mean error, which calculates how far each data point is from its assigned centroid. 
        #the smaller the error, the tighter the cluster. 
        error = np.sum((cdist(self.centroids, self.centroids[self.labels])**2)) / len(self.labels)
        return error



    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise ValueError("model has not been fitted yet.")
        return self.centroids
