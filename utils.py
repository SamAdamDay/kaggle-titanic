import pandas as pd
import numpy as np
import math
import copy

class Error(BaseException):
    pass

class DataHandler():
    """A container for handling and preprocessing data.
    
    Parameters
    ----------
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
        full training set into training and evaluation sets)."""

    def __init__(self, seed=None):
        self.N_full_train = None
        self.N_test = None
        self.N_eval = None
        self.N_test = None
        self.full_train = None
        self.test = None
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
        
        test_fraction specifies the proportion of data going into the test
        set."""

        # Compute the size of the splitted sets
        self.N_train = math.floor(self.N_full_train * 0.9)
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

    def _update_data(self, new_full_train, new_test, inplace=False):
        """Update the full train and test data, either in place or copying."""

        if inplace:
            instance = self
        else:
            instance = copy.copy(self)

        instance.full_train = new_full_train
        instance.test = new_test

        return instance

    def make_dummies(self, columns=None, inplace=False):
        """Convert categorical data columns into dummy variables"""

        new_full_train = pd.get_dummies(self.full_train, columns=columns)
        new_test = pd.get_dummies(self.test, columns=columns)

        return self._update_data(new_full_train, new_test, inplace)


class DataHandlerTitantic(DataHandler):

    def to_is_female(self, inplace=False):
        """Convert the 'Sex' column into an int 'IsFemale' column."""

        if "Sex" not in self.full_train.columns:
            raise Error("Already converted 'Sex' to 'IsFemale'")

        replace_dict = {"Sex": {"female": 1, "male": 0}}
        rename_dict = {"Sex": "IsFemale"}

        new_full_train = self.full_train.replace(replace_dict)
        new_full_train = new_full_train.rename(columns=rename_dict)

        new_test = self.test.replace(replace_dict)
        new_test = new_test.rename(columns=rename_dict)

        return self._update_data(new_full_train, new_test, inplace)