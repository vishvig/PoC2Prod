import math
import numpy as np

from sklearn.cluster import KMeans
from scipy.stats import ks_2samp
from scipy.stats import multinomial


class Score(object):
    def __init__(self):
        pass

    @staticmethod
    def log_cluster_metric(X, Xn):
        """
        Check representation of synthetic data in realtion to original data
        :param X: Original dataset
        :param Xn: Synthetic dataset
        :return: Score
        """
        Xm = np.concatenate((X, Xn[0]), axis=0)
        Na = X.shape[0]
        Nb = Xn[0].shape[0]
        clusterer = KMeans(n_clusters=3)
        merged_labels = clusterer.fit_predict(Xm)
        c = Na / (Na + Nb)

        unique_m, counts_m = np.unique(merged_labels, return_counts=True)
        unique_o, counts_o = np.unique(merged_labels[0:Na], return_counts=True)

        _sum = 0

        for _index, i in enumerate(unique_m):
            _sum += ((counts_m[_index] / counts_o[_index]) - c) ** 2
        score = math.log(_sum / len(unique_m))
        return score

    def kstest(self, X, Xn):
        multinomial.pmf()
