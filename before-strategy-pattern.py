from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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


def train_pipeline(algorithm: str) -> None:
    # get Iris data in DataFrame
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # fit the algorithm
    if algorithm == "logistic_regression":
        model = LogisticRegression()
        # scale features on X
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # fit to logistic regression
        model.fit(X_train, y_train)

    elif algorithm == "gradient_boosting":
        model = LGBMClassifier(n_estimators=50)
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
    else:
        raise NotImplementedError

    # get training evaluation score
    y_train_pred = model.predict(X_train)
    train_score = accuracy_score(y_train, y_train_pred)

    # get test evaluation score
    y_test_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_test_pred)

    # print train and test evaluation score
    print(
        f"{algorithm} | train score: {train_score:.5f} | test score: {test_score:.5f}"
    )


if __name__ == "__main__":
    algorithm1 = "logistic_regression"
    train_pipeline(algorithm1)
    algorithm2 = "gradient_boosting"
    train_pipeline(algorithm2)
