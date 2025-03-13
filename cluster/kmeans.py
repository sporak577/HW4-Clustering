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

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
