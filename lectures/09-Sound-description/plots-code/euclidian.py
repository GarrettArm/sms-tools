import numpy as np
import matplotlib.pyplot as plt


def eucDist(vec1, vec2):
    return np.sqrt(np.sum(np.power(np.array(vec1) - np.array(vec2), 2)))
