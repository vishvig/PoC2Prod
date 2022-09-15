import time

import numpy as np
import pandas as pd

from kmodes.kmodes import KModes
from kneed import KneeLocator
from scipy.stats import dirichlet, multinomial

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class MNMM(object):
    def __init__(self, K=None, max_clusters=10, rtol=1e-3, max_iter=100, restarts=10, random_seed=1729):
        self.set_K = False
        if K is None:
            self.set_K = True
        self._K = K
        self.clusters = range(1, max_clusters + 1)
        self.max_clusters = max_clusters
        self._rtol = rtol
        self._max_iter = max_iter
        self._restarts = restarts
        self.seed = random_seed

        self.loss = [-float('inf')]
        self.alpha = [list()]
        self.beta = [list()]
        self.gamma = [list()]
        self.label_encoders = list()
        self.one_hot_encoders = list()
        self.columns = list()

    def _find_best_k(self, X, clusters):
        """
        Find the best K, number of components
        :param X:
        :return:
        """
        cost = list()
        if clusters > self.max_clusters:
            k_range = range(1, self.max_clusters + 1)
        else:
            k_range = range(1, clusters + 2)
        print(k_range)

        for k in k_range:
            kmm = KModes(n_clusters=k)
            kmm.fit(X)
            print(kmm.cost_)
            cost.append(kmm.cost_)
        # print(cost)
        self._K = KneeLocator(k_range,
                              cost,
                              curve="convex",
                              direction="decreasing",
                              S=0,
                              interp_method="interp1d").elbow
        print(self._K)

    def fit(self, X):
        """
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.

        :param X: (N, C), matrix of counts
        :return:
        """
        n_samples, n_features = X.shape
        self.loss = self.loss * n_features
        self.alpha = self.alpha * n_features
        self.beta = self.beta * n_features
        self.gamma = self.gamma * n_features
        self.columns = list(X.columns)
        classes = 0

        start_time = time.time()
        x_discrete_cat = np.zeros((n_samples, len(self.columns)))
        for col_index, col in enumerate(self.columns):
            # x_discrete_cat = np.zeros((n_samples, 1))

            enc = LabelEncoder()
            enc.fit(X[col])
            # x_discrete_cat[:, 0] = enc.transform(X[col])
            x_discrete_cat[:, col_index] = enc.transform(X[col])
            self.label_encoders.append(enc)
            if len(enc.classes_) > classes:
                classes = len(enc.classes_)

        if self.set_K:
            self._find_best_k(x_discrete_cat, clusters=classes)
        print(self._K)
        end_time = time.time()
        print(f"Time to calculate K: {end_time - start_time}")
        for col_index, col in enumerate(self.columns):
            print(f'Processing Col {col}')
            oh_enc = OneHotEncoder(handle_unknown='ignore')
            x_sparse = oh_enc.fit_transform(x_discrete_cat[:, col_index].reshape(-1, 1)).toarray()
            self.one_hot_encoders.append(oh_enc)

            for it in range(self._restarts):
                # print('iteration %i' % it)
                alpha, beta, gamma, loss = self._train_once(x_sparse)

                if len(self.alpha[col_index]) == 0:
                    self.alpha[col_index] = alpha
                if len(self.beta[col_index]) == 0:
                    self.beta[col_index] = beta
                if len(self.gamma[col_index]) == 0:
                    self.gamma[col_index] = gamma

                if loss > self.loss[col_index]:
                    # print('better loss on iteration %i: %.10f' % (it, loss))
                    self.loss[col_index] = loss
                    self.alpha[col_index] = alpha
                    self.beta[col_index] = beta
                    self.gamma[col_index] = gamma

                else:
                    break
        # print('-----------')
        return self

    def _init_params(self, X):
        C = X.shape[1]
        weights = np.random.randint(1, 20, self._K)
        alpha = weights / weights.sum()
        beta = dirichlet.rvs([2 * C] * C, self._K)
        return alpha, beta

    def _train_once(self, X):
        """
        Runs one full cycle of the EM algorithm
        :param X: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        """
        loss = float('inf')
        alpha, beta = self._init_params(X)
        # print(f'beta: {beta}')
        gamma = None

        for i, it in enumerate(range(self._max_iter)):
            # print(f'----iter {i}')
            prev_loss = loss
            gamma = self._e_step(X, alpha, beta)
            # print(f'gamma: {gamma}')
            alpha, beta = self._m_step(X, gamma)
            # print(f'weights: {alpha}, {beta}')
            loss = self._compute_vlb(X, alpha, beta, gamma)
            # print('Loss: %f' % loss)
            # print(f'--------')
            if it > 0 and np.abs((prev_loss - loss) / prev_loss) < self._rtol:
                break
        # print(f'beta after EM: {beta}')
        return alpha, beta, gamma, loss

    @staticmethod
    def _multinomial_prob(counts, beta, log=False):
        """
        Evaluates the multinomial probability for a given vector of counts
        counts: (N x C), matrix of counts
        beta: (C), vector of multinomial parameters for a specific cluster k
        Returns:
        p: (N), scalar values for the probabilities of observing each count vector given the beta parameters
        """
        n = counts.sum(axis=-1)
        # print(f"===beta: {beta}")
        m = multinomial(n, beta)
        # print(f'===m: {m}')
        if log:
            return m.logpmf(counts)
        return m.pmf(counts)

    def _compute_vlb(self, X, alpha, beta, gamma):
        """
        X: (N x C), matrix of counts
        alpha: (K)  mixture component weights
        beta: (K x C), multinomial categories weights
        gamma: (N x K), posterior probabilities for cluster assignments
        :return:  The variational lower bound value
        """
        loss = 0
        for k in range(beta.shape[0]):
            weights = gamma[:, k]
            loss += np.sum(weights * (np.log(alpha[k]) + self._multinomial_prob(X, beta[k], log=True)))
            loss -= np.sum(weights * np.log(weights))
        return loss

    def _e_step(self, X, alpha, beta):
        """
        Performs E-step on MNMM model
        X: (N x C), matrix of counts
        alpha: (K) mixture component weights
        beta: (K x C), multinomial categories weights
        Returns:
        gamma: (N x K), posterior probabilities for objects clusters assignments
        """
        # Compute gamma
        N = X.shape[0]
        K = beta.shape[0]
        weighted_multi_prob = np.zeros((N, K))
        for k in range(K):
            weighted_multi_prob[:, k] = alpha[k] * self._multinomial_prob(X, beta[k])
        # print(f'weighted_multi_prob: {weighted_multi_prob[:, 0]}')
        # To avoid division by 0
        weighted_multi_prob[weighted_multi_prob == 0] = np.finfo(float).eps
        # print(f'weighted_multi_prob: {weighted_multi_prob[:, 0]}')

        denum = weighted_multi_prob.sum(axis=1)
        gamma = weighted_multi_prob / denum.reshape(-1, 1)

        return gamma

    @staticmethod
    def _m_step(X, gamma):
        """
        Performs M-step on MNMM model
        X: (N x C), matrix of counts
        gamma: (N x K), posterior probabilities for objects clusters assignments
        Returns:
        alpha: (K), mixture component weights
        beta: (K x C), mixture categories weights
        """
        # Compute alpha
        alpha = gamma.sum(axis=0) / gamma.sum()

        # Compute beta
        weighted_counts = gamma.T.dot(X)
        beta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)

        return alpha, beta

    def sample(self, n=100):
        rng = np.random.default_rng(seed=self.seed)
        xn_dict = dict()
        for col_i, col in enumerate(self.columns):
            values = list()
            # print(self.alpha[col_i])
            cluster_n = rng.multinomial(n, self.alpha[col_i])
            # print(cluster_n)
            classes = self.label_encoders[col_i].classes_
            # print(classes)
            # print(self.beta[col_i])
            for i, cluster in enumerate(cluster_n):
                values.extend(list(
                    np.random.choice(np.arange(0, len(classes)), size=cluster,
                                     p=self.beta[col_i][i])))
            xn_dict[col] = self.label_encoders[col_i].inverse_transform(values)
        synthetic_df = pd.DataFrame(xn_dict, columns=self.columns)
        return synthetic_df

    def _loss(self):
        return self.loss

    def _alpha(self):
        return self.alpha

    def _beta(self):
        return self.beta

    def _gamma(self):
        return self.gamma
