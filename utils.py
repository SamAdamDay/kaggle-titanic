import pandas as pd
import numpy as np
import math

class Error(BaseException):
    pass

class DataHandler():
    """A container for handling and preprocessing data.

    Keeps track of all change in the data from preprocessing, for debugging
    purposes.
    
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
        full training set into training and validation sets).

    N_valid
        The number of data points in the validation set (after splitting the
        full training set into training and validation sets).
        
    _history_full_train
        A list of the history of the full training set.
        
    _history_test
        A list of the history of the test set."""

    def __init__(self, seed=None):
        self.N_full_train = None
        self.N_test = None
        self.N_valid = None
        self.N_test = None
        self._history_full_train = []
        self._history_test = []
        self._rng = np.random.default_rng(seed)
        self._train_indices = None
        self._valid_indices = None

    def load_train_data(self, path):
        """Loads training data from the csv file at path."""
        self._history_full_train = [pd.read_csv("data/train.csv")]
        self.N_full_train = len(self._history_full_train[0].index)

    def load_test_data(self, path):
        """Loads test data from the csv file at path."""
        self._history_test = [pd.read_csv("data/test.csv")]
        self.N_test = len(self._history_test[0].index)

    def shuffle_split(self, test_fraction=0.9):
        """Shuffle the data and split into test and valid sets.
        
        test_fraction specifies the proportion of data going into the test
        set."""

        # Compute the size of the splitted sets
        self.N_train = math.floor(self.N_full_train * 0.9)
        self.N_valid = self.N_full_train - self.N_train

        # A permuter for the full training set, for use when dynamically
        # accessing the split train and valid sets
        permuter_full_train = self._rng.permutation(self.N_full_train)

        # The set of indices for train and validation data sets
        self._train_indices = permuter_full_train[:self.N_train]
        self._valid_indices = permuter_full_train[self.N_train:]

    def _update_full_train(self, new_data):
        """Update the full training data to new_data"""
        self._history_full_train.append(new_data)

    def _update_test(self, new_data):
        """Update the test data to new_data"""
        self._history_test.append(new_data)

    @property
    def data_full_train(self):
        """Get (modified) full training data"""

        if len(self._history_full_train) == 0:
            raise AttributeError("Full training data not yet loaded.")

        return self._history_full_train[-1]

    @property
    def data_test(self):
        """Get (modified) test data"""

        if len(self._history_test) == 0:
            raise AttributeError("Test data not yet loaded.")

        return self._history_test[-1]

    @property
    def train(self):
        """Get the (modified) splitted train data set"""

        if len(self._history_full_train) == 0:
            raise AttributeError("Full training data not yet loaded.")
        if self._train_indices is None:
            raise AttributeError("Data not yet split into train and valid"
                                 " sets")
        
        
        return self.data_full_train.loc[self._train_indices]

    @property
    def valid(self):
        """Get the (modified) splitted validation data set"""

        if len(self._history_full_train) == 0:
            raise AttributeError("Full training data not yet loaded.")
        if self._valid_indices is None:
            raise AttributeError("Data not yet split into train and valid"
                                 " sets")
        
        return self.data_full_train.loc[self._valid_indices]

    def make_dummies(self, columns=None):
        """Convert categorical data columns into dummy variables"""

        new_full_train = pd.get_dummies(self.data_full_train, columns=columns)
        self._update_full_train(new_full_train)
        new_test = pd.get_dummies(self.data_test, columns=columns)
        self._update_test(new_test)


class DataHandlerTitantic(DataHandler):

    def to_is_female(self):
        """Convert the 'Sex' column into an int 'IsFemale' column."""

        if "Sex" not in self.data_full_train.columns:
            raise Error("Already converted 'Sex' to 'IsFemale'")

        replace_dict = {"Sex": {"female": 1, "male": 0}}
        self._update_full_train(self.data_full_train.replace(replace_dict))
        self._update_test(self.data_test.replace(replace_dict))