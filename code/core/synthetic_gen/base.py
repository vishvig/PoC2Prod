import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from .gmm import GMM
from .bmm import BMM
from .score import Score


class SyntheticDataGen(object):
    def __init__(self):
        self.model = dict()
        self.encs = dict()

    def to_numpy(self, df, columns):
        """
        Convert dataframe to numpy array by converting categorical variables as integer labels
        :param df: The dataframe to convert
        :param columns: The continuous & discrete columns
        :return: Converted numpy arrays of X & Y and the label encoders
        """
        # Fetch the columns from the dataframe
        _columns = list(df.columns)

        # declaring empty arrays for the predictor and target variables
        X = np.empty(df[[i for k, v in columns.items() for i in v]].shape)

        # Assign the values from the columns, that are declared continuous,
        # into the corresponding positions in the numpy array
        for i, col in enumerate(columns['continuous']):
            X[:, _columns.index(col)] = df[col]

        # Encoding for X (predictors)
        for i, col in enumerate(columns['categorical']):
            # Initialize, fit and store the label encoder for column
            enc = LabelEncoder()
            enc.fit(df[col])
            self.encs[col] = enc

            # Transform the column using the encoder object
            X[:, _columns.index(col)] = enc.transform(df[col])

        return X

    def fit(self, data, discrete=None, ids=None):
        """

        :param data:
        :param discrete:
        :param ids:
        :return:
        """
        if discrete is None:
            discrete = list()
        if ids is None:
            ids = list()
        cols_val = list()
        cols_val.extend(discrete)
        cols_val.extend(ids)

        if discrete is None and ids is None:
            continuous_df = data
        else:
            continuous_df = data.loc[:, ~data.columns.isin(cols_val)]
        gmm = GMM()
        gmm.fit(continuous_df)
        self.model['continuous'] = gmm
        # print(self.model)

        if len(discrete) > 0:
            discrete_df = data.loc[:, data.columns.isin(discrete)]
            bmm = BMM()
            bmm.fit(discrete_df)
            self.model['discrete'] = bmm
            # print(self.model)

    def _score(self, data, Xn, method='log_cluster_metric'):
        score = getattr(Score, method)[data, Xn]
        return score

    def sample(self, n=100):
        """
        Generate synthetic data
        :return:
        """
        final_df_list = list()
        if 'continuous' in self.model:
            xn_cnt = self.model['continuous'].sample(n=n)
            final_df_list.append(xn_cnt)
        if 'discrete' in self.model:
            xn_dscrt = self.model['discrete'].sample(n=n)
            final_df_list.append(xn_dscrt)
        Xn = pd.concat(final_df_list, axis=1)
        return Xn
