import pickle
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import shap

from cyc_pep_perm.data.paths import MODEL_RF_RANDOM_DW, TRAIN_RANDOM_DW_

PARAMS = {
    "n_estimators": [100, 200, 300, 400, 500],  # number of trees
    "max_features": ["sqrt", "log2", 1.0],  # features to consider at every split
    "max_depth": [5, 10, 20, 30],  # maximum depth of tree
    "min_samples_split": [2, 5, 10, 20],  # min samples required to split a node
    "min_samples_leaf": [1, 2, 4, 8],  # min samples required to be at a leaf node
    "bootstrap": [True, False],  # method of selecting samples for training each tree
}


class RFRegressor:
    """
    Class for training and evaluating a random forest regressor.
    """

    def __init__(self):
        self.datapath = None
        self.data = None
        self.X = None
        self.y = None
        self.best_model = None

    def train(
            self,
            datapath=TRAIN_RANDOM_DW_,
            params=PARAMS,
            savepath=MODEL_RF_RANDOM_DW
            ):
        # Data
        self.datapath = datapath
        assert os.path.exists(self.datapath), "File does not exist"
        self.data = pd.read_csv(self.datapath)
        self.X = self.data.drop(["SMILES", "target"], axis=1)
        self.y = self.data["target"]

        # Model
        model = RandomForestRegressor()

        # K-fold cross validation
        kf = KFold(n_splits=5, shuffle=True)

        # Gridsearch
        gs = GridSearchCV(
            model,
            params,
            cv=kf,
            # scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        gs.fit(self.X, self.y)

        self.best_model = gs.best_estimator_

        print(f"Best parameters: {gs.best_params_}")

        # save best model
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, "wb") as f:
            pickle.dump(self.best_model, f)

        return self.best_model

    def evaluate(self, X=None, y=None):
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

    def predict(self, X):
        assert self.best_model is not None, "Best model not found - load or train model"
        y_pred = self.best_model.predict(X)
        return y_pred

    def load(self, modelpath=MODEL_RF_RANDOM_DW):
        assert os.path.exists(modelpath), "File does not exist"
        with open(modelpath, "rb") as f:
            self.best_model = pickle.load(f)
        return self.best_model

    def test(self, testpath):
        assert os.path.exists(testpath), "File does not exist"
        test_data = pd.read_csv(testpath)
        X_test = test_data.drop(["SMILES", "target"], axis=1)
        y_test = test_data["target"]
        assert self.best_model is not None, "Best model not found - load or train model"
        y_pred = self.best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"RMSE: {rmse:.3f}")
        print(f"R-squared: {r2:.3f}")
        return y_pred, rmse, r2

    def shap_explain(self, X=None):
        assert self.best_model is not None, "Best model not found - load or train model"
        if X is None:
            X = self.X
        assert self.X is not None, "Data not found - load or train model"
        explainer = shap.Explainer(self.best_model)
        shap_values = explainer(self.X)

        shap.summary_plot(shap_values)
        return shap_values
