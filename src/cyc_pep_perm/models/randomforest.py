import os
import pickle
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold


PARAMS = {
    "n_estimators": [100, 200, 300, 400, 500],  # number of trees
    # "max_features": ["sqrt", "log2", 1.0],  # features to consider at every split
    # "max_depth": [5, 10, 20, 30],  # maximum depth of tree
    # "min_samples_split": [2, 5, 10, 20],  # min samples required to split a node
    # "min_samples_leaf": [1, 2, 4, 8],  # min samples required to be at a leaf node
    # "bootstrap": [True, False],  # method of selecting samples for training each tree
}


class RF:
    """
    A class used to represent a random forest regressor model.

    Attributes:
        datapath (str): The path to the training data.
        data (pandas.DataFrame): The training data.
        X (pandas.DataFrame): The features of the training data.
        y (pandas.Series): The target variable of the training data.
        best_model (sklearn.ensemble.RandomForestRegressor): The best trained random
        forest regressor model.

    """

    def __init__(self):
        """
        The constructor for RFRegressor class.
        """
        self.datapath: str = None
        self.data: pd.DataFrame = None
        self.X: pd.DataFrame = None
        self.y: pd.Series = None
        self.best_model: RandomForestRegressor = None

    def train(
        self,
        datapath: Union[str, pd.DataFrame],
        savepath: str,
        params: Dict[str, List[Any]] = PARAMS,
        seed: int = 42,
        n_folds: int = 8,
    ) -> RandomForestRegressor:
        """
        Trains a random forest regressor model.

        Args:
            datapath (str): The path to the training data.
            savepath (str): The
            path to save the trained model.
            params (Dict[str, list]): The
            hyperparameters for the random forest regressor model.

        Returns:
            RandomForestRegressor: The best trained random forest regressor model.

        Raises:
            AssertionError: If the specified datapath does not exist.
        """
        # Set seed
        np.random.seed(seed)

        # Data
        self.datapath = datapath
        if isinstance(self.datapath, pd.DataFrame):
            self.data = self.datapath
        else:
            assert os.path.exists(self.datapath), "File does not exist"
            self.data = pd.read_csv(self.datapath)
        self.X = self.data.drop(["SMILES", "target"], axis=1)
        self.y = self.data["target"]

        # Model
        model = RandomForestRegressor()

        # K-fold cross validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        # Gridsearch
        gs = GridSearchCV(
            model,
            params,
            cv=kf,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        gs.fit(self.X, self.y)

        self.best_model = gs.best_estimator_

        print(f"Best parameters: {gs.best_params_}")

        # save best model
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, "wb") as f:
            pickle.dump(self.best_model, f)
        print(f"Best model saved to {savepath}")

        return self.best_model

    def evaluate(self, X: pd.DataFrame = None, y: pd.Series = None) -> tuple:
        """
        Evaluates the trained model on given data.

        Args:
            X (pandas.DataFrame, optional): The features of the data to evaluate. If not
            provided, uses the training data.
            y (pandas.Series, optional): The target
            variable of the data to evaluate. If not provided, uses the training data.

        Returns:
            tuple: A tuple containing the predicted values, RMSE (Root Mean Squared
            Error), and R-squared score.

        Raises:
            AssertionError: If the best model is not found (not loaded or trained).

        """

        assert self.best_model is not None, "Best model not found - load or train model"
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        # Evaluation metrics
        y_pred = self.best_model.predict(self.X)
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        r2 = r2_score(self.y, y_pred)

        print(f"RMSE: {rmse:.3f}")
        print(f"R-squared: {r2:.3f}")

        return y_pred, rmse, r2

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Args:
            X (pandas.DataFrame): The features of the data to make predictions.

        Returns:
            numpy.ndarray: The predicted values.

        Raises:
            AssertionError: If the best model is not found (not loaded or trained).

        """

        assert self.best_model is not None, "Best model not found - load or train model"
        y_pred = self.best_model.predict(X)
        return y_pred

    def load(self, modelpath: str) -> RandomForestRegressor:
        """
        Loads a trained model from a file.

        Args:
            modelpath (str): The path to the trained model file.

        Returns:
            sklearn.ensemble.RandomForestRegressor: The loaded trained model.

        Raises:
            AssertionError: If the specified modelpath does not exist.

        """

        assert os.path.exists(modelpath), "File does not exist"
        with open(modelpath, "rb") as f:
            self.best_model = pickle.load(f)
        return self.best_model

    def test(self, testpath: Union[str, pd.DataFrame]) -> tuple:
        """
        Evaluates the trained model on a test dataset.

        Args:
            testpath (str): The path to the test dataset.

        Returns:
            tuple: A tuple containing the predicted values, RMSE (Root Mean Squared
            Error), and R-squared score.

        Raises:
            AssertionError: If the specified testpath does not exist. AssertionError: If
            the best model is not found (not loaded or trained).

        """
        if isinstance(testpath, pd.DataFrame):
            test_data = testpath
        else:
            assert os.path.exists(testpath), "File does not exist"
            test_data = pd.read_csv(testpath)
        self.X_test = test_data.drop(["SMILES", "target"], axis=1)
        self.y_test = test_data["target"]
        assert self.best_model is not None, "Best model not found - load or train model"
        y_pred = self.best_model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        print(f"RMSE: {rmse:.3f}")
        print(f"R-squared: {r2:.3f}")
        return y_pred, rmse, r2

    def shap_explain(self, X: np.ndarray = None) -> np.ndarray:
        """
        Generates SHAP (SHapley Additive exPlanations) values for the trained model.

        Args:
            X (pandas.DataFrame, optional): The features of the data to generate SHAP
            values. If not provided, uses the training data.

        Returns:
            numpy.ndarray: The SHAP values.

        Raises:
            AssertionError: If the best model is not found (not loaded or trained).
            AssertionError: If the training data is not found (not loaded or trained).

        """

        assert self.best_model is not None, "Best model not found - load or train model"
        if X is None:
            X = self.X
        assert X is not None, "Data not found - load or train model"
        explainer = shap.Explainer(self.best_model)
        shap_values = explainer(X)

        shap.summary_plot(shap_values)
        return shap_values
