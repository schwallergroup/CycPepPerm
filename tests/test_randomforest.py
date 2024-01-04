import unittest
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from cyc_pep_perm.models.randomforest import RF
from cyc_pep_perm.data.paths import TRAIN_RANDOM_DW


class TestRandomForest(unittest.TestCase):
    def setUp(self):
        self.rf = RF()
        self.datapath = TRAIN_RANDOM_DW
        df = pd.read_csv(self.datapath)
        self.X = df.drop(["SMILES", "target"], axis=1)
        self.y = df["target"]
        filepath = os.path.dirname(__file__)
        self.savepath = os.path.join(filepath, 'test_models', "rf_unittest.pkl")
        self.rf.train(datapath=self.datapath, savepath=self.savepath)

    def test_train(self):
        self.assertIsInstance(self.rf.best_model, RandomForestRegressor)

    def test_evaluate(self):
        self.rf.load(self.savepath)
        y_pred, rmse, r2 = self.rf.evaluate(self.X, self.y)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(r2, float)

    def test_predict(self):
        self.rf.load(self.savepath)
        y_pred = self.rf.predict(self.X)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_load(self):
        loaded_model = self.rf.load(self.savepath)
        self.assertIsInstance(loaded_model, RandomForestRegressor)

    def test_test(self):
        self.rf.load(self.savepath)
        y_pred, rmse, r2 = self.rf.test(self.datapath)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(r2, float)

    def test_shap_explain(self):
        self.rf.load(self.savepath)
        shap_values = self.rf.shap_explain(self.X)
        self.assertIsInstance(shap_values.values, np.ndarray)


if __name__ == '__main__':
    unittest.main()
