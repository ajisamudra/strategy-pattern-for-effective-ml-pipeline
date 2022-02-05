from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Any

# We will demonstrate how we would create ML pipeline for training different learning algorithms without and with using Strategy Pattern

# Problem Statement
# In this example, we will use the Iris dataset which consists of 3 different types of flowers
# Hence we will frame the problem as multi-class classification problem
# We will use two different library for this purpose: sklearn and lightgbm
# Even though they have similar interface (`fit` and `predict`),
# they have different input arguments for the same method in `fit` method

# sklearn linear_model.LogisticRegression `fit` method has 3 attributes fit(X, y, sample_weight=None)
# while, lightgbm.LGBMClassifier `fit` method has >3 attributes some of them are
# fit(X, y, sample_weight=None, eval_set=None, eval_names=None, eval_metric=None, early_stopping_rounds=None)
# With those additional arguments in lightbgm we have flexibility to prevent overfitting in training process
# The training will stop if the eval_score is not improving after a certain number of rounds

# On the other hand, LogisticRegression will also require us to scale the data before calling the `fit` method
# This is because the LogisticRegression is a linear model and it is sensitive to the different range values on the data
# And the scaling will also help fasten the training process to reach the optimal solution

# So this kind of situation highly probable to happen in real world application
# The question is how would we handle this situation?
# We will exercise to create ML pipeline starting without using Strategy Pattern
# and then we will use Strategy Pattern to create a pipeline with the same functionality, but with much more efficient way

# First how we will crate a pipeline that will be able to train different learning algorithms?
# We might use if-else to acommodate two or more different algorithms

from abc import ABC, abstractmethod


class LearningAlgorithm(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        # the implementation will be defined in derived class
        pass

    @abstractmethod
    def predict(self, X_test):
        # the implementation will be defined in derived class
        pass


class SklearnLogReg(LearningAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        self.__model = LogisticRegression(*args, **kwargs)
        self.__scaler = StandardScaler()

    def fit(self, X_train, y_train):
        # fit scaler to features on X_train
        X_train = self.__scaler.fit_transform(X_train)
        # fit to logistic regression
        self.__model.fit(X_train, y_train)

    def predict(self, X_test):
        # scale features on X_test
        X_test = self.__scaler.transform(X_test)
        return self.__model.predict(X_test)  # predict


class LgbmClassifier(LearningAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        self.__model = LGBMClassifier(*args, **kwargs)

    def fit(self, X_train, y_train):
        # split the X_train again to get X_val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=123
        )
        # fit to gradient boosting
        # with early stopping
        # and eval_set to prevent overfitting
        self.__model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(0),
            ],
        )

    def predict(self, X_test):
        return self.__model.predict(X_test)  # predict


def train_pipeline(algorithm: LearningAlgorithm) -> None:
    # get Iris data in DataFrame
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # fit the algorithm
    model = algorithm
    model.fit(X_train, y_train)

    # get training evaluation score
    y_train_pred = model.predict(X_train)
    train_score = accuracy_score(y_train, y_train_pred)

    # get test evaluation score
    y_test_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_test_pred)

    # print train and test evaluation score
    print(
        f"{type(algorithm).__name__} | train score: {train_score:.5f} | test score: {test_score:.5f}"
    )


if __name__ == "__main__":
    algorithm1 = SklearnLogReg()
    train_pipeline(algorithm1)
    algorithm2 = LgbmClassifier()
    train_pipeline(algorithm2)
    algorithm3 = "logistic_regression"
    train_pipeline(algorithm3)  # this will throw error
