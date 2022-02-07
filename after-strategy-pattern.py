from abc import ABC, abstractmethod
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LearningAlgorithm(ABC):
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # the implementation will be defined in derived class
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        # the implementation will be defined in derived class
        pass


class SklearnLogReg(LearningAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        self.__model = LogisticRegression(*args, **kwargs)
        self.__scaler = StandardScaler()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # fit scaler to features on X_train
        X_train = self.__scaler.fit_transform(X_train)
        # fit to logistic regression
        self.__model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        # scale features on X_test
        X_test = self.__scaler.transform(X_test)
        return self.__model.predict(X_test)


class LgbmClassifier(LearningAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        self.__model = LGBMClassifier(*args, **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # split the X_train again to get X_val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=123
        )
        # fit to gradient boosting
        self.__model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(0),
            ],
        )

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.__model.predict(X_test)


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
    algorithm2 = LgbmClassifier(n_estimators=50)
    train_pipeline(algorithm2)
    algorithm3 = "logistic_regression"
    train_pipeline(algorithm3)  # this will throw error
