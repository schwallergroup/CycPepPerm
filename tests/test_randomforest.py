import os
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from cyc_pep_perm.models import RF

# make tet data so independent of the data file
columns = [
    "SMILES",
    "target",
    "MW",
    "cLogP",
    "cLogS",
    "HBA",
    "HBD",
    "Total Surface Area",
    "Rel. PSA",
    "PSA",
    "Rot. Bonds",
    "Amides",
]
data = [
    [
        "CC(C)[C@@H](C(NCCSCc1cccc(CSCCC(N[C@@H](CC(NCCOCCOCCCCCCCl)=O)C(NCc2cc3ccc2)=O)=O)n1)=O)NC3=O",
        41.206681848330696,
        821.5,
        3.0452,
        -6.508,
        13.0,
        5.0,
        644.43,
        0.29243,
        227.45,
        15.0,
        5.0,
    ],
    [
        "O=C(C[C@@H](C(N[C@H](CCCC1)[C@@H]1C(N(CCC1)C[C@H]1C(NCCSCc1cccc(CSCC2)n1)=O)=O)=O)NC2=O)NCCOCCOCCCCCCCl",
        39.74796989085007,
        825.53,
        2.9212,
        -6.027,
        13.0,
        4.0,
        640.81,
        0.28174,
        218.66,
        14.0,
        5.0,
    ],
    [
        "O=C(C[C@@H](C(NCC(NC(CC1)CCC1CC(NCCSCc1cccc(CSCC2)n1)=O)=O)=O)NC2=O)NCCOCCOCCCCCCCl",
        24.527463251624283,
        785.47,
        2.1131,
        -5.843,
        13.0,
        5.0,
        615.45,
        0.3062,
        227.45,
        14.0,
        5.0,
    ],
    [
        "OC[C@@H](C(NCCSCc1cccc(CSCCC(N[C@@H](CC(NCCOCCOCCCCCCCl)=O)C(N[C@H]2Cc3c[nH]cn3)=O)=O)n1)=O)NC2=O",
        13.128624730382676,
        813.44,
        -0.312,
        -4.362,
        16.0,
        7.0,
        628.35,
        0.36048,
        276.36,
        17.0,
        5.0,
    ],
    [
        "CC(C)[C@@H](C(NCCSCc1cccc(CSCCC(N[C@@H](CC(NCCOCCOCCCCCCCl)=O)C(N[C@H]2Cc3ccccc3)=O)=O)n1)=O)NC2=O",
        86.64762792782881,
        835.53,
        3.0732,
        -6.46,
        13.0,
        5.0,
        656.93,
        0.28686,
        227.45,
        17.0,
        5.0,
    ],
    [
        "OC[C@@H](C(NC(CC1)CCC1CC(NCCSCc1cccc(CSCCC(N[C@H]2CC(NCCOCCOCCCCCCCl)=O)=O)n1)=O)=O)NC2=O",
        26.662660910881982,
        815.49,
        1.5457,
        -5.714,
        14.0,
        6.0,
        634.3,
        0.31775,
        247.68,
        15.0,
        5.0,
    ],
    [
        "C[C@@H](C(NCCSCc1cccc(CSCCC(N[C@@H](CC(NCCOCCOCCCCCCCl)=O)C(N[C@H]2[C@H]3CCCC2)=O)=O)n1)=O)NC3=O",
        71.64447241968833,
        785.47,
        2.018,
        -5.951,
        13.0,
        5.0,
        612.69,
        0.30758,
        227.45,
        14.0,
        5.0,
    ],
]

DF = pd.DataFrame(data, columns=columns)


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
        self.df = DF
        self.X = self.df.drop(["SMILES", "target"], axis=1)
        self.y = self.df["target"]
        filepath = os.path.dirname(__file__)
        self.savepath = os.path.join(filepath, "test_models", "rf_unittest.pkl")
        self.rf.train(datapath=self.df, savepath=self.savepath, n_folds=2)

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
        y_pred, rmse, r2 = self.rf.test(self.df)
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
