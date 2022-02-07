from lightgbm import LGBMClassifier, log_evaluation, early_stopping
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_pipeline(
    algorithm: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:

    # fit the algorithm
    if algorithm == "logistic_regression":
        model = LogisticRegression()
        # scale features on X
        scaler = StandardScaler()
        transformed_X_train = scaler.fit_transform(X_train)
        transformed_X_test = scaler.transform(X_test)
        # fit to logistic regression
        model.fit(transformed_X_train, y_train)

        # get training evaluation score
        y_train_pred = model.predict(transformed_X_train)
        train_score = accuracy_score(y_train, y_train_pred)

        # get test evaluation score
        y_test_pred = model.predict(transformed_X_test)
        test_score = accuracy_score(y_test, y_test_pred)

    elif algorithm == "gradient_boosting":
        model = LGBMClassifier()
        # split the X_train again to get X_val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=123
        )
        # fit to gradient boosting
        # with early stopping
        # and eval_set to prevent overfitting
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(0),
            ],
        )
        # get training evaluation score
        y_train_pred = model.predict(X_train)
        train_score = accuracy_score(y_train, y_train_pred)

        # get test evaluation score
        y_test_pred = model.predict(X_test)
        test_score = accuracy_score(y_test, y_test_pred)

    else:
        raise NotImplementedError

    # print train and test evaluation score
    print(
        f"{algorithm} | train score: {train_score:.5f} | test score: {test_score:.5f}"
    )


if __name__ == "__main__":
    # get Iris data in DataFrame
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    algorithm1 = "logistic_regression"
    train_pipeline(algorithm1, X_train, X_test, y_train, y_test)
    algorithm2 = "gradient_boosting"
    train_pipeline(algorithm2, X_train, X_test, y_train, y_test)
    algorithm3 = "other_algorithm"  # this will throw error
    train_pipeline(algorithm3, X_train, X_test, y_train, y_test)
