import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold


class GMM(object):
    def __init__(self, random_state=1729):
        self.rs = random_state

        self.model = None
        self.columns = list()

        self.trainNLL = []
        self.validNLL = []
        self.hyperparams = []
        self.valid_max_iter = None
        self.valid_reg_covar = None

        self.default_max_iters = [10, 100, 1000, 10000, 100000, 1000000]
        self.default_reg_covars = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        self.default_n_components = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50)
        self.default_cov_types = ['full', 'spherical', 'diag', 'tied']

    def get_best_model(self, X):
        min_nll_index = self.validNLL.index(min(self.validNLL))
        hyperparams = self.hyperparams[min_nll_index].split('-')
        cov_type = self.default_cov_types[int(hyperparams[0])]
        components = self.default_n_components[int(hyperparams[1])]
        print(f"Best hyperparameters: {cov_type}-{components}")
        self.model = BayesianGaussianMixture(n_components=components, covariance_type=cov_type, random_state=self.rs,
                                             max_iter=self.valid_max_iter, reg_covar=self.valid_reg_covar)
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
                    train_scores.append(np.sum(-model.score_samples(Xtrain)) / Xtrain.shape[0])
                    val_scores.append(np.sum(-model.score_samples(Xval)) / Xval.shape[0])
            except ValueError as e:
                raise ValueError(e)
            except ConvergenceWarning as e:
                raise ConvergenceWarning(e)
        return {'train_score': train_scores, 'test_score': val_scores}

    def fit(self, X):
        self.columns = X.columns
        X = X.to_numpy()
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            for reg_covar in self.default_reg_covars:
                if self.valid_reg_covar is None:
                    self.valid_reg_covar = reg_covar
                else:
                    print(f"valid reg_covar from training: {self.valid_reg_covar}")
                    break
                print(f"Trying reg_covar: {reg_covar}")
                for max_iter in self.default_max_iters:
                    if self.valid_max_iter is None:
                        self.valid_max_iter = max_iter
                    else:
                        if self.valid_reg_covar is None:
                            continue
                        if len(self.hyperparams) > 0:
                            print(f"valid max_iter from training: {self.valid_max_iter}")
                            break
                    print(f"Trying max_iter: {max_iter}")
                    for i, cov in enumerate(self.default_cov_types):
                        if self.valid_max_iter is None or self.valid_reg_covar is None:
                            break
                        for j, n in enumerate(self.default_n_components):
                            if self.valid_max_iter is None or self.valid_reg_covar is None:
                                break
                            print(f"{i}-{j}")
                            try:
                                model = BayesianGaussianMixture(n_components=n,
                                                                covariance_type=cov,
                                                                random_state=self.rs,
                                                                max_iter=self.valid_max_iter,
                                                                reg_covar=self.valid_reg_covar
                                                                )
                                scores = self.cross_validation(model, X, cv=5)
                                self.trainNLL.append(np.mean(scores['train_score']))
                                self.validNLL.append(np.mean(scores['test_score']))
                                self.hyperparams.append(f"{i}-{j}")
                            except ValueError as e:
                                warnings.warn(f"reg_covar {reg_covar} is too small : {e}")
                                self.valid_reg_covar = None
                                self.valid_max_iter = None
                                self.trainNLL = []
                                self.validNLL = []
                                self.hyperparams = []
                                break
                            except ConvergenceWarning:
                                warnings.warn(f"max_iter {max_iter} is too small to reach convergence")
                                self.valid_max_iter = None
                                self.trainNLL = []
                                self.validNLL = []
                                self.hyperparams = []
                                break
        self.get_best_model(X=X)

    def sample(self, n=100):
        xnew = self.model.sample(n)[0]
        Xn = pd.DataFrame(xnew, columns=self.columns)
        return Xn
