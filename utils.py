# THis file contains utilities functions for our codebase 
import numpy as np 

def euclid_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2) ** 2 ))
