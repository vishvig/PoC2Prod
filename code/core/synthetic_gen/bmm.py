import pandas as pd
import numpy as np
import warnings
from scipy.stats import dirichlet
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold


class BernoulliMixture(object):
    def __init__(self, n_components, max_iter=100, random_state=1729, verbose=False, tol=1e-3):
        self.K = n_components
        self.max_iter = max_iter
        self.rs = random_state
        self.verbose = verbose
        self.tol = tol

        self.weights = None
        self.means = None

        self.converged = False
        self.n_samples, self.n_features = None, None
        self.enc = None
        self.warn_division_by_zero_resp = False
        self.warn_division_by_mstep = False

    @staticmethod
    def bernoulli_vectorized(X, means):
        """
        To compute the probability of x for each bernoulli distribution
        data = N X D matrix
        means = K X D matrix
        prob (result) = N X K matrix
        """
        try:
            N = X.shape[0]
            F = X.shape[1]

            # compute prob(x/mean)
            # prob[i, k] for ith data point, and kth cluster/mixture distribution
            Xn = X.reshape((N, 1, F))
            Xn_o = (1 - X).reshape((N, 1, F))
            mu = np.tile(means, (N, 1, 1))
            mu_o = np.tile((1 - means), (N, 1, 1))
            _bernoulli = np.multiply(np.power(mu, Xn), np.power(mu_o, Xn_o))
            prob = np.prod(_bernoulli, axis=2)
            log_pmf = np.log(_bernoulli.clip(min=1e-50))
            return prob, log_pmf
        except Exception as e:
            raise Exception(e)

    @staticmethod
    def bernoulli(X, means):
        """
        To compute the probability of x for each bernoulli distribution
        data = N X D matrix
        means = K X D matrix
        prob (result) = N X K matrix
        """
        try:
            N = X.shape[0]
            F = X.shape[1]
            K = len(means)
            # compute prob(x/mean)
            # prob[i, k] for ith data point, and kth cluster/mixture distribution
            prob = np.zeros((N, K))
            log_pmf = np.zeros((N, K, F))
            for i in range(N):
                _bernoulli = ((means ** X[i]) * ((1 - means) ** (1 - X[i])))
                prob[i, :] = np.prod(_bernoulli, axis=1)
                log_pmf[i] = np.log(_bernoulli.clip(min=1e-50))
            return prob, log_pmf
        except Exception as e:
            raise Exception(e)

    def compute_resp(self, X, weights, means):
        """
        To compute responsibilities, or posterior probability p(z/x)
        data = N X D matrix
        weights = K dimensional vector
        means = K X D matrix
        prob or resp (result) = N X K matrix
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            # step 1
            # calculate the p(x/means)
            prob, log_pmf = self.bernoulli(X, means)

            # step 2
            # calculate the numerator of the resp.s
            prob = prob * weights

            # step 3
            # calcualte the denominator of the resp.s
            row_sums = prob.sum(axis=1)[:, np.newaxis]

            # step 4
            # calculate the resp.s
            try:
                prob = prob / row_sums
                return prob, log_pmf
            except ZeroDivisionError:
                if not self.warn_division_by_zero_resp:
                    warnings.warn("Division by zero occurred in responsibility calculations!")
                    self.warn_division_by_zero_resp = True
                raise ZeroDivisionError
            except RuntimeWarning:
                if not self.warn_division_by_zero_resp:
                    warnings.warn("Division by zero occurred in responsibility calculations!")
                    self.warn_division_by_zero_resp = True
                raise RuntimeWarning
            except Exception as e:
                warnings.warn(f'{e}')
                raise Exception

    def e_step(self, X, weights, means):
        """
        To compute expectation of the loglikelihood of Mixture of Bernoulli distributions
        Since computing E(LL) requires computing responsibilities, this function does a double-duty
        to return responsibilities too
        """
        try:
            resp, log_pmf = self.compute_resp(X, weights, means)
            ll = resp * (np.log(weights) + np.sum(log_pmf, axis=2))
            return ll, resp
        except RuntimeWarning:
            return None, None
        except Exception as e:
            warnings.warn(f"{e}")
            return None, None

    def m_step(self, X, resp):
        """
        Re-estimate the parameters using the current responsibility
        data = N X D matrix
        resp = N X K matrix
        return revised weights (K vector) and means (K X D matrix)
        """
        N = len(X)
        D = len(X[0])
        K = len(resp[0])

        Nk = np.sum(resp, axis=0)
        mus = np.empty((K, D))

        # print(resp[:, 0][:, np.newaxis])
        for k in range(K):
            mus[k] = np.sum(resp[:, k][:, np.newaxis] * X, axis=0)  # sum is over N data points
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    mus[k] = mus[k] / Nk[k]
                except ZeroDivisionError:
                    if not self.warn_division_by_mstep:
                        warnings.warn("Division by zero occurred in Mixture of Bernoulli Dist M-Step!")
                        self.warn_division_by_mstep = True
                    break
                except RuntimeWarning:
                    if not self.warn_division_by_mstep:
                        warnings.warn("Division by zero occurred in Mixture of Bernoulli Dist M-Step!")
                        self.warn_division_by_mstep = True
                    break
        return Nk / N, mus

    def em_algorithm(self, X, weights, means):
        """
        EM algo for Mixture of Bernoulli Distributions
        """
        ll, resp = self.e_step(X, weights, means)
        likelihood = np.sum(ll)
        ll_old = likelihood

        for i in range(self.max_iter):
            if self.verbose and (i % 5 == 0):
                print("iteration {}:".format(i))
                print("   {}:".format(weights))
                print("   {:.6}".format(likelihood))

            # E Step: calculate resps
            # Skip, rolled into log likelihood calc
            # For 0th step, done as part of initialization

            # M Step
            weights, means = self.m_step(X, resp)

            # convergence check
            # print(f"Performing E-step")
            ll, resp = self.e_step(X, weights, means)
            likelihood = np.sum(ll)
            # print(np.abs(likelihood - ll_old))
            if np.abs(likelihood - ll_old) < self.tol:
                # print("Relative gap:{:.8} at iterations {}".format(likelihood - ll_old, i))
                self.converged = True
                break
            else:
                ll_old = likelihood
        if not self.converged:
            warnings.warn(f"Failed to converge", ConvergenceWarning)
        self.weights, self.means = weights, means

    def init_params(self, X):
        D = X.shape[1]
        init_wts = np.random.uniform(.25, .75, self.K)
        tot = np.sum(init_wts)
        init_wts = init_wts / tot
        init_means = dirichlet.rvs([2 * D] * D, self.K, random_state=self.rs)
        return init_wts[:], init_means[:]

    def fit(self, X):
        """
        Picks N random points of the selected 'digits' from MNIST data set and
        fits a model using Mixture of Bernoulli distributions.
        And returns the weights and means.
        """
        self.converged = False
        self.n_samples, self.n_features = X.shape
        self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        x_sparse = self.enc.fit_transform(X)

        # initalize
        weights, means = self.init_params(X=x_sparse)
        self.em_algorithm(X=x_sparse, weights=weights, means=means)

    def sample(self, n=100):
        rng = np.random.default_rng(seed=self.rs)
        # xn_dict = dict()
        Xn = np.zeros((n, self.n_features))
        cluster_n = rng.multinomial(n, self.weights)
        start_row = 0
        end_row = 0
        for i, cluster in enumerate(cluster_n):
            end_row += cluster
            cat_col = 0
            for j, category in enumerate(self.enc.categories_):
                num_classes = len(category.tolist())
                Xn[start_row:end_row, j] = np.random.choice(category, size=cluster,
                                                            p=self.means[i][cat_col:cat_col + num_classes])
                cat_col += num_classes
            start_row += cluster
        return Xn

    def score_samples(self, X):
        x_sparse = self.enc.transform(X)
        # print(self.weights)
        # print(self.means)
        ll, resp = self.e_step(x_sparse, self.weights, self.means)
        return ll


class BMM(object):
    def __init__(self, random_state=1729):
        self.rs = random_state

        self.model = None
        self.columns = list()
        self.encs = dict()

        self.trainNLL = []
        self.validNLL = []
        self.hyperparams = []
        self.valid_max_iter = None
        self.valid_tol = None

        self.default_max_iters = [100, 1000, 10000]
        self.default_tols = [1e-1]
        # self.default_n_components = (2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50)
        self.default_n_components = (2, 3, 5, 10, 20, 30, 50)

    def to_numpy(self, df):
        # declaring empty arrays for the predictor and target variables
        # X = np.empty(df[[i for k, v in columns for i in v]].shape)
        X = np.empty(df.shape)

        # Encoding for X (predictors)
        for i, col in enumerate(self.columns):
            # Initialize, fit and store the label encoder for column
            enc = LabelEncoder()
            enc.fit(df[col])
            self.encs[col] = enc

            # Transform the column using the encoder object
            # X[:, _columns.index(col)] = enc.transform(df[col])
            X[:, i] = enc.transform(df[col])

        return X

    def get_best_model(self, X):
        min_nll_index = self.validNLL.index(min(self.validNLL))
        hyperparams = self.hyperparams[min_nll_index]
        components = self.default_n_components[int(hyperparams[0])]
        print(f"Best hyperparameters: {components}")
        self.model = BernoulliMixture(n_components=components, random_state=self.rs,
                                      max_iter=self.valid_max_iter, tol=self.valid_tol)
        self.model.fit(X)

    def cross_validation(self, model, X, cv=5):
        train_scores = list()
        val_scores = list()
        _cv = KFold(n_splits=cv, random_state=self.rs, shuffle=True)
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            try:
                for train_index, val_index in _cv.split(X):
                    Xtrain, Xval = X[train_index], X[val_index]
                    model.fit(Xtrain)
                    train_score = model.score_samples(Xtrain)
                    val_score = model.score_samples(Xval)
                    if train_score is not None:
                        train_scores.append(np.sum(-train_score) / Xtrain.shape[0])
                    if val_score is not None:
                        val_scores.append(np.sum(-val_score) / Xval.shape[0])
            except ValueError as e:
                raise ValueError(e)
            except ConvergenceWarning as e:
                raise ConvergenceWarning(e)
        return {'train_score': train_scores, 'test_score': val_scores}

    def fit(self, X):
        self.columns = list(X.columns)
        X = self.to_numpy(X)
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            for tol in self.default_tols:
                if self.valid_tol is None:
                    self.valid_tol = tol
                else:
                    print(f"valid tol from training: {self.valid_tol}")
                    break
                print(f"Trying tol: {tol}")
                for max_iter in self.default_max_iters:
                    if self.valid_max_iter is None:
                        self.valid_max_iter = max_iter
                    else:
                        if self.valid_tol is None:
                            continue
                        if len(self.hyperparams) > 0:
                            print(f"valid max_iter from training: {self.valid_max_iter}")
                            break
                    print(f"Trying max_iter: {max_iter}")
                    for i, n in enumerate(self.default_n_components):
                        if self.valid_max_iter is None or self.valid_tol is None:
                            break
                        print(f"{i}")
                        try:
                            model = BernoulliMixture(n_components=n,
                                                     random_state=self.rs,
                                                     max_iter=self.valid_max_iter,
                                                     tol=self.valid_tol, verbose=False
                                                     )
                            scores = self.cross_validation(model, X, cv=5)
                            if None in scores['train_score']:
                                scores['train_score'].remove(None)
                            if None in scores['test_score']:
                                scores['test_score'].remove(None)
                            if len(scores['test_score']) == 0 or len(scores['train_score']) == 0:
                                warnings.warn(f'Cross-validation returned None for {n} clusters.'
                                              f' Trying more clusters will not yield better results. Ending run')
                                break
                            self.trainNLL.append(np.mean(scores['train_score']))
                            self.validNLL.append(np.mean(scores['test_score']))
                            self.hyperparams.append(f"{i}")
                        except ConvergenceWarning:
                            warnings.warn(f"max_iter {max_iter} is too small to reach convergence")
                            self.valid_max_iter = None
                            self.trainNLL = []
                            self.validNLL = []
                            self.hyperparams = []
                            break
        self.get_best_model(X=X)

    def sample(self, n=100):
        syn_dict = dict()
        xnew = self.model.sample(n)
        for k, v in self.encs.items():
            _x = xnew[:, self.columns.index(k)].astype(int).tolist()
            syn_dict[k] = v.inverse_transform(_x)
        # Xn = pd.DataFrame(xnew, columns=self.columns)
        Xn = pd.DataFrame(syn_dict)
        return Xn
