import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from torch.utils.data import Dataset
import torch
import math
import copy

class Error(BaseException):
    pass

class DataHandler():
    """A container for handling and preprocessing data.
    
    Parameters
    ----------
    target_column
        The column of the full train data which is the target.

    seed
        A seed for the random number generator.
        
    Attributes
    ----------
    N_full_train
        The number of data points in the full training set.

    N_test
        The number of data points in the test set.

    N_train
        The number of data points in the training set (after splitting the
        full training set into training and evaluation sets).

    N_eval
        The number of data points in the evaluation set (after splitting the
        full training set into training and evaluation sets).
    """

    def __init__(self, target_column=None, seed=None):
        self.N_full_train = None
        self.N_test = None
        self.N_eval = None
        self.N_test = None
        self.full_train = None
        self.test = None
        self.target_column = target_column
        self._rng = np.random.default_rng(seed)
        self._train_indices = None
        self._eval_indices = None

    def load_data(self, train_path, test_path):
        """Loads training test data from csv files."""
        self.full_train = pd.read_csv("data/train.csv")
        self.test = pd.read_csv("data/test.csv")
        self.N_full_train = len(self.full_train.index)
        self.N_test = len(self.test.index)

    def shuffle_split(self, test_fraction=0.9):
        """Shuffle the data and split into test and eval sets.
        
        Parameters
        ----------
        test_fraction
            Specifies the proportion of data going into the test set.
        """

        # Compute the size of the splitted sets
        self.N_train = math.floor(self.N_full_train * test_fraction)
        self.N_eval = self.N_full_train - self.N_train

        # A permuter for the full training set, for use when dynamically
        # accessing the split train and eval sets
        permuter_full_train = self._rng.permutation(self.N_full_train)

        # The set of indices for train and evaluation data sets
        self._train_indices = permuter_full_train[:self.N_train]
        self._eval_indices = permuter_full_train[self.N_train:]

    @property
    def train(self):
        """Get the (modified) splitted train data set"""

        if len(self.full_train) == 0:
            raise AttributeError("Full training data not yet loaded.")
        if self._train_indices is None:
            raise AttributeError("Data not yet split into train and eval"
                                 " sets")
        
        
        return self.full_train.loc[self._train_indices]

    @property
    def eval(self):
        """Get the (modified) splitted evaluation data set"""

        if len(self.full_train) == 0:
            raise AttributeError("Full training data not yet loaded.")
        if self._eval_indices is None:
            raise AttributeError("Data not yet split into train and eval"
                                 " sets")
        
        return self.full_train.loc[self._eval_indices]

    def get_train_pytorch_dataset(self, feature_columns):
        """Create a PyTorchDataSet for the train data set."""
        
        return PyTorchDataSet(self.train, feature_columns, 
                              self.target_column)

    def get_eval_pytorch_dataset(self, feature_columns):
        """Create a PyTorchDataSet for the eval data set."""
        
        return PyTorchDataSet(self.eval, feature_columns, self.target_column)

    def _make_new_instance(self, new_full_train, new_test):
        """Copy the `self`, updating the data."""

        new_instance = copy.copy(self)

        new_instance.full_train = new_full_train
        new_instance.test = new_test

        return new_instance

    def impute_values(self, columns=None, *, missing_values=np.nan, 
                      strategy="mean", fill_value=None, n_neighbors=5, 
                      weights="uniform", metric="nan_euclidean",
                      add_indicator=False, fit_test_data=False):
        """Impute missing values in `columns`

        Parameters
        ----------
        columns : array_like or None, default=None
            The columns to impute, and to use for fitting the imputer.
        
        missing_values : int, float, str, np.nan or None, default=np.nan
            The placeholder for the missing values.
            
        strategy : str, default="mean" 
            The strategy to use for imputation.

            - "mean" : replace with the mean for that column.
            - "median" : replace with the median for that column.
            - "most_frequent" : replace with the mode for that column.
            - "constant" : replace with the constant value `fill_value`.
            - "knn" : use the k-neareast neighbours imputer KNNImputer, from
              scikit-learn.

        fill_value : str or None, default=None
            When strategy == "constant", fill_value is used to replace all
            occurrences of missing_values. If left to the default, fill_value
            will be 0 when imputing numerical data and "missing_value" for 
            strings or object data types.

        n_neighbors : int, default=5
            When strategy == "knn", this is the number of neighboring samples
            to use for imputation.

        weights : {'uniform', 'distance'} or callable, default='uniform'
            When strategy == "knn", this is the weight function used in
            prediction. Possible values:

            - 'uniform' : uniform weights. All points in each neighborhood are
              weighted equally.
            - 'distance' : weight points by the inverse of their distance.
              in this case, closer neighbors of a query point will have a
              greater influence than neighbors which are further away.
            - callable : a user-defined function which accepts an
              array of distances, and returns an array of the same shape
              containing the weights.

        metric : {'nan_euclidean'} or callable, default='nan_euclidean'
            When strategy == "knn", this is the distance metric for searching
            neighbours. Possible values:

            - 'nan_euclidean'
            - callable : a user-defined function which conforms to the 
              definition of ``_pairwise_callable(X, Y, metric, **kwds)``. The
              function accepts two arrays, X and Y, and a `missing_values`
              keyword in `kwds` and returns a scalar distance value.

        add_indicator : bool, default=False
            If True, a :class:`MissingIndicator` transform will stack onto the
            output of the imputer's transform. This allows a predictive
            estimator to account for missingness despite imputation. If a
            feature has no missing values at fit/train time, the feature won't
            appear on the missing indicator even if there are missing values
            at transform/test time.

        fit_test_data : bool, default=False
            If set to True, will fit the imputer with all of the data
            (including test data). Otherwise, will only use self.train

        Returns
        -------
        imputed_data_handler : DataHandler
            Returns the :class:`DataHandler` instance with missing values
            replaces.
        """
        
        # The columns of the data to impute
        columns_train = self.train[columns]
        columns_full_train = self.full_train[columns]
        columns_test = self.test[columns]

        # Select the imputer based on `strategy`
        if strategy in ("mean", "median", "most_frequent", "constant"):
            imputer = SimpleImputer(missing_values=missing_values,
                                    strategy=strategy, fill_value=fill_value,
                                    add_indicator=add_indicator)
        elif strategy == "knn":
            imputer = KNNImputer(missing_values=missing_values, 
                                 n_neighbors=n_neighbors, weights=weights,
                                 metric=metric, add_indicator=add_indicator)
        else:
            raise ValueError(f"Unknown imputer strategy: {strategy!r}")

        # Fit the imputer using the just the training data
        if fit_test_data:
            imputer.fit(pd.concat(columns_full_train, columns_test))
        else:
            imputer.fit(columns_train)

        # Now add the missing values
        new_columns_full_train = imputer.transform(columns_full_train)
        new_columns_test = imputer.transform(columns_test)

        # Replace the columns in the data, and add the new columns (if they
        # exist)
        new_full_train = self.full_train.copy()
        new_test = self.test.copy()
        new_full_train[columns] = new_columns_full_train
        new_test[columns] = new_columns_test

        return self._make_new_instance(new_full_train, new_test)

    def make_dummies(self, columns=None):
        """Convert categorical data `columns` into dummy variables."""

        new_full_train = pd.get_dummies(self.full_train, columns=columns)
        new_test = pd.get_dummies(self.test, columns=columns)

        return self._make_new_instance(new_full_train, new_test)

    def expand_polyonimially(self, columns=None, degree=2, 
                             include_bias=False):
        """Generate polynomial features for `columns`."""

        # Extract the relevant columns
        columns_full_train = self.full_train[columns]
        columns_test = self.test[columns]

        # The polynomial feature generator
        poly_feat = PolynomialFeatures(degree, include_bias=include_bias)

        # Polynomalially expand the columns
        new_columns_full_train = poly_feat.fit_transform(columns_full_train)
        new_columns_test = poly_feat.fit_transform(columns_test)

        # Get the new column names
        new_columns = poly_feat.get_feature_names_out(columns)

        # Copy the data, replacing the old columns with the new
        new_full_train = self.full_train.drop(columns=columns)
        new_test = self.test.drop(columns=columns)
        new_full_train[new_columns] = new_columns_full_train
        new_test[new_columns] = new_columns_test

        return self._make_new_instance(new_full_train, new_test)
        


class DataHandlerTitantic(DataHandler):

    def __init__(self, seed=None):
        super().__init__(target_column="Survived", seed=seed)

    def to_is_female(self):
        """Convert the 'Sex' column into an int 'IsFemale' column."""

        if "Sex" not in self.full_train.columns:
            raise Error("Already converted 'Sex' to 'IsFemale'")

        replace_dict = {"Sex": {"female": 1, "male": 0}}
        rename_dict = {"Sex": "IsFemale"}

        new_full_train = self.full_train.replace(replace_dict)
        new_full_train = new_full_train.rename(columns=rename_dict)

        new_test = self.test.replace(replace_dict)
        new_test = new_test.rename(columns=rename_dict)

        return self._make_new_instance(new_full_train, new_test)


class PyTorchDataSet(Dataset):
    """"A data set object for use with PyTorch."""

    def __init__(self, data, feature_columns, target_column):

        # Get the columns as numpy arrays
        X_numpy = data[feature_columns].to_numpy()
        y_numpy = data[target_column].to_numpy()

        # Make tensors out of them
        self.X = torch.tensor(X_numpy).float()
        self.y = torch.tensor(y_numpy)

        # If the target column is an integer type, assume that it has
        # categorical data. PyTorch needs this to be a long type.
        if isinstance(y_numpy.dtype, np.integer):
            self.y = self.y.long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]