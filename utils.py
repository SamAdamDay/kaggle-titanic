import pandas as pd
import numpy as np
import math

rng = np.random.default_rng()

class DataHandler():
    """A container for handling and preprocessing data.
    
    Parameters
    ----------
    seed
        A seed for the random number generator."""

    def __init__(self, seed=None):
        self.data_full_train = None
        self.data_test = None
        self.N_full_train = None
        self.N_test = None
        self.N_valid = None
        self.N_test = None
        self.permuter_full_train = None
        self.rng = np.random.default_rng(seed)

    def load_train_data(self, path):
        """Loads training data from the csv file at path."""
        self.data_full_train = pd.read_csv("data/train.csv")
        self.N_full_train = len(self.data_full_train.index)

    def load_test_data(self, path):
        """Loads test data from the csv file at path."""
        self.data_test = pd.read_csv("data/train.csv")
        self.N_test = len(self.data_test.index)

    def shuffle_split(self, test_fraction=0.9):
        """Shuffle the data and split into test and valid sets.
        
        test_fraction specifies the proportion of data going into the test
        set."""

        # Compute the size of the splitted sets
        self.N_train = math.floor(self.N_full_train * 0.9)
        self.N_valid = self.N_full_train - self.N_train

        # A permuter for the full training set, for use when dynamically
        # accessing the split train and valid sets
        self.permuter_full_train = self.rng.permutation(self.N_full_train)

    @property
    def train(self):
        """Get the splitted train data set"""

        if self.data_full_train is None:
            raise AttributeError("Full training data not yet loaded.")
        if self.permuter_full_train is None:
            raise AttributeError("Data not yet split into train and valid"
                                 " sets")
        
        train_indices = self.permuter_full_train[:self.N_train]
        return self.data_full_train.loc[train_indices]

    @property
    def valid(self):
        """Get the splitted validation data set"""

        if self.data_full_train is None:
            raise AttributeError("Full training data not yet loaded.")
        if self.permuter_full_train is None:
            raise AttributeError("Data not yet split into train and valid"
                                 " sets")
        
        valid_indices = self.permuter_full_train[self.N_train:]
        return self.data_full_train.loc[valid_indices]


class DataHandlerTitantic(DataHandler):
    """A container for handling and preprocessing the Titanic data.
    
    Parameters
    ----------
    seed
        A seed for the random number generator."""

    pass