Kaggle Titanic Competition
==========================

This contains my solutions to the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).


Methodology
-----------

- I start out with the simplest models, gradually increasing in complexity (either in terms of number of features considered, or choice of model).
- The 'training data' I call the 'full training data'. I split this into training data and evaluation data. This split is the same for all models.
- I use the evaluation data to select the best model.
- To generate the prediction for the test set, I first retrain the selected model on the full training set.
- I abstract the preprocessing steps to the `DataHandler` container, defined in [utils.py](./utils.py), which makes sure that the same steps are applied to all data splits, and that there is no data leakage.


Repo structure
--------------

- [logistic-regression.ipynb](./logistic-regression.ipynb): using logistic regression and some feature creation. I fit a logistic regression model with inceasing number of features.
- [neural-network.ipynb](./neural-network.ipynb): using feedforward neural networks. I test out various neural network architectures, optimisers and learning rates.
- [utils.py](./utils.py): houses the `DataHandler` container.
    * Holds the full training and test data. 
    * Splits the full training data into train and evaluation sets.
    * Has various preprocessing methods, which make sure that the same transformations apply to all the splits of the data, and that there is no leakage.
