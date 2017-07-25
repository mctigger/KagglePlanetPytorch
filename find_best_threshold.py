import numpy as np
from sklearn.metrics import fbeta_score, make_scorer
import itertools
import pathos.multiprocessing


def fbeta(true_label, prediction):
   return fbeta_score(true_label, prediction, beta=2, average='samples')


def optimise_f2_thresholds_fast(y, p, iterations=100, verbose=True):
    best_threshold = [0.2]*17
    for t in range(17):
        best_fbeta = 0
        temp_threshhold = [0.2]*17
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(y, p > temp_threshhold)
            if temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshold[t] = temp_value

        if verbose:
            print(t, best_fbeta, best_threshold[t])

    return best_threshold


def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = [0.2] * 17
    for i in range(17):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)

    return x