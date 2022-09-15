import warnings

import numpy as np
import pandas as pd
from scipy.stats import dirichlet, multinomial

from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import ConvergenceWarning


class MNMM(object):
    def __init__(self, n_components=2, rtol=1e-3, max_iter=100, random_state=1729):
        self._K = n_components
        self._rtol = rtol
        self._max_iter = max_iter
        self.seed = random_state

        self.loss = [-float('inf')]
        self.alpha = [list()]
        self.beta = [list()]
        self.gamma = [list()]
        # self.label_encoders = list()
        self.one_hot_encoders = list()
        self.converged = False
        # self.columns = list()

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
        # self.columns = list(X.columns)
        # classes = 0

        # for col_index, col in enumerate(self.columns):
        for col_index, col in enumerate(range(n_features)):
            # print(f'Processing Col {col}')
            oh_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            oh_enc.fit(X[:, col_index].reshape(-1, 1))
            self.one_hot_encoders.append(oh_enc)
            x_sparse = oh_enc.transform(X[:, col_index].reshape(-1, 1))

            # print('iteration %i' % it)
            alpha, beta, gamma, loss = self._train_once(x_sparse)

            if len(self.alpha[col_index]) == 0:
                self.alpha[col_index] = alpha
            if len(self.beta[col_index]) == 0:
                self.beta[col_index] = beta
            if len(self.gamma[col_index]) == 0:
                self.gamma[col_index] = gamma
        # print('-----------')
        return self

    def _init_params(self, X):
        C = X.shape[1]
        weights = np.random.randint(1, 20, self._K)
        alpha = weights / weights.sum()
        beta = dirichlet.rvs([2 * C] * C, self._K, self.seed)
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
                self.converged = True
                break
        if self.converged is False:
            warnings.warn(f"Failed to converge", ConvergenceWarning)
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

    def score_samples(self, X):
        rng = np.random.default_rng(seed=self.seed)
        n_samples, n_features = X.shape
        likelihood = np.zeros((1, n_features))
        for col_index, col in enumerate(range(n_features)):
            cluster_n = rng.multinomial(n_samples, self.alpha[col_index])
            sum_n = 0
            for i, n in enumerate(cluster_n):
                sum_n += n
                if i == 0:
                    x = X[0:sum_n, col_index].reshape(-1, 1)
                else:
                    x = X[cluster_n[i - 1]:sum_n, col_index].reshape(-1, 1)
                try:
                    oh_enc = self.one_hot_encoders[col_index]
                    x_sparse = oh_enc.transform(x)
                    # n = x_sparse.sum(axis=0)
                    m = multinomial(n, self.beta[col_index][i])
                    trail_x = x_sparse.sum(axis=0)
                    # log_pmf = m.logpmf(trail_x).sum()
                    log_pmf = m.pmf(trail_x).sum()
                    likelihood[:, col_index] += log_pmf
                except Exception as e:
                    pass
            # likelihood[:, col_index] = self._multinomial_prob(x_sparse,
            #                                                   beta=self.beta[col_index],
            #                                                   log=True)
        return likelihood

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

    def _converged(self):
        return self.converged
