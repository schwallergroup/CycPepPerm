import os
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from cyc_pep_perm.data.paths import TRAIN_RANDOM_DW
from cyc_pep_perm.models.randomforest import RF


class TestRandomForest(unittest.TestCase):
    """
    Unit tests for the RandomForest class.
    """

    def setUp(self):
        """
        Set up the test environment by initializing the RF object, loading the data,
        and training the random forest model.

        Parameters:
        - None

        Returns:
        - None
        """
        self.rf = RF()
        self.datapath = TRAIN_RANDOM_DW
        df = pd.read_csv(self.datapath)
        self.X = df.drop(["SMILES", "target"], axis=1)
        self.y = df["target"]
        filepath = os.path.dirname(__file__)
        self.savepath = os.path.join(filepath, "test_models", "rf_unittest.pkl")
        self.rf.train(datapath=self.datapath, savepath=self.savepath)

    def test_train(self):
        """
        Test case to verify that the best_model attribute of the RandomForest instance
        is an instance of RandomForestRegressor.
        """
        self.assertIsInstance(self.rf.best_model, RandomForestRegressor)

    def test_evaluate(self):
        """
        Test the evaluate method of the RandomForest class.

        This method tests the evaluate method of the RandomForest class by loading a
        trained model, evaluating it on the given input data, and checking the types of
        the returned values.

        Returns:
            None
        """
        self.rf.load(self.savepath)
        y_pred, rmse, r2 = self.rf.evaluate(self.X, self.y)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(r2, float)

    def test_predict(self):
        """
        Test the predict method of the RandomForest class.

        This method loads a trained random forest model, makes predictions on the input
        data, and checks if the predicted values are of type numpy.ndarray.

        Returns:
            None
        """
        self.rf.load(self.savepath)
        y_pred = self.rf.predict(self.X)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_load(self):
        """
        Test the load method of the RandomForest class.

        This method ensures that the loaded model is an instance of
        RandomForestRegressor.
        """
        loaded_model = self.rf.load(self.savepath)
        self.assertIsInstance(loaded_model, RandomForestRegressor)

    def test_test(self):
        """
        Test the `test` method of the RandomForest class.

        This method loads a trained random forest model, performs predictions on a test
        dataset, and asserts the types of the predicted values, root mean squared error
        (rmse), and coefficient of determination (r2).

        Returns:
            None
        """
        self.rf.load(self.savepath)
        y_pred, rmse, r2 = self.rf.test(self.datapath)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(r2, float)

    def test_shap_explain(self):
        """
        Test the `shap_explain` method of the RandomForest class.

        This method tests whether the `shap_explain` method returns an instance of
        `np.ndarray`.
        """
        self.rf.load(self.savepath)
        shap_values = self.rf.shap_explain(self.X)
        self.assertIsInstance(shap_values.values, np.ndarray)


if __name__ == "__main__":
    unittest.main()
