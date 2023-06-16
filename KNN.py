# ---------------------------------------------------------
# IMPLIMENTING THE KNN CLASS IN CODE 
# ---------------------------------------------------------

# importing libraries 
from utils import euclid_dist 
from collections import Counter 

# importing modules
import numpy as np

# BASE CLASS 
class KNN :
    '''
    This class implements the KNN Module and computes the classification.
    Parameters : 
    k = int (hyper parameter which implies no of neighbours to check for )
    '''
    def __init__(self, k:int) -> None:
        self.k = k 

    def fit(self, X, y):
        print("---------- Fitting the KNN model for the data ---------------")
        self.X_train, self.y_train = X, y 
        print("Your X varible size: ", X.shape)
        print("Your y variable size: ", y.shape)

    def predict(self, X) :
        ''' Predicts the labels of each model based on k(hyperparamter) nearest neighbours '''
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x) :
        ''' creates a prediction using the K Nearest Neighbours -> Helper function for predict function '''
        # compute the distances (Euclidean Distance)
        distances = [euclid_dist(x, x_train) for x_train in self.X_train]
        # get k nearest neighbours /samples and labels 
        k_indices = np.argsort(distances)[0:self.k] # indices of k nearest points
        k_nearest_labels = [self.y_train[i] for i in k_indices] # lables of k nearest indices
        # majority vote / most common class label
        most_common = Counter(k_nearest_labels).most_common(1) 
        return most_common[0][0]

