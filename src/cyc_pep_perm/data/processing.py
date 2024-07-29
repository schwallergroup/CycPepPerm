import os
import pickle
from typing import Optional

import pandas as pd
from mordred import Calculator, descriptors
from pandas_ods_reader import read_ods
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

from cyc_pep_perm.data import FEATURES_DW, MORDRED_DESCS


class DataProcessing:
    """
    Class for processing data related to cyclic peptide membrane permeability.
    """

    def __init__(
        self,
        datapath: str,
        target_label: str = "CAPA [1 µM]",
        smiles_label: str = "SMILES",
        data: Optional[pd.DataFrame] = None,
        df_mordred: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the DataProcessing object.

        Args:
            datapath (str): Path to the data file.
            target_label (str): Label of the target variable.
            smiles_label (str): Label of the SMILES column.
        """
        self.datapath = datapath
        assert os.path.exists(self.datapath), "File does not exist"
        self.data_dir = os.path.dirname(self.datapath)
        self.target_label = target_label
        self.smiles_label = smiles_label
        self.data = data
        self.df_mordred = df_mordred
        self.calculator = Calculator(descriptors, ignore_3D=True)
        print(f"Target column: {self.target_label}")
        print(f"SMILES column: {self.smiles_label}")

    def read_data(self, filename: Optional[str] = None):
        """
        Read the data from the file and perform necessary preprocessing.
        """
        try:
            self.data = pd.read_csv(self.datapath)
        except Exception:
            self.data = read_ods(self.datapath, 1)

        self.smiles = self.data[self.smiles_label]
        self.mols = [Chem.MolFromSmiles(smile) for smile in self.smiles]

        if self.target_label in self.data.columns:
            self.data.rename(columns={self.target_label: "target"}, inplace=True)
        else:
            print(f"No target column {self.target_label} found.")
        self.data.rename(columns={self.smiles_label: "SMILES"}, inplace=True)

        # drop irrelevant columns for training
        if "target" in self.data.columns:
            self.data = self.data[["SMILES", "target"] + FEATURES_DW]
        else:
            self.data = self.data[["SMILES"] + FEATURES_DW]

        if filename:
            new_filepath = os.path.join(self.data_dir, filename)
        else:
            new_filepath = os.path.join(
                self.data_dir, f'{self.datapath.split("/")[-1].split(".")[0]}.csv'
            )
        self.data.to_csv(new_filepath, index=False)

        print(f"Saved data to {new_filepath}")

        return self.data

    def calc_mordred(self, filename: Optional[str] = None):
        """
        Calculate Mordred descriptors for the molecules and save the results to a file.

        Args:
            filename (str, optional): Path to save the Mordred data. If not provided, a
            default filename will be used.
        """
        if self.data is None:
            self.read_data()

        self.df_mordred = self.calculator.pandas(self.mols)
        self.df_mordred = self.df_mordred[MORDRED_DESCS]

        self.df_mordred["SMILES"] = self.smiles
        if self.data is not None and isinstance(self.data, pd.DataFrame):
            if "target" in self.data.columns:
                self.df_mordred["target"] = self.data["target"]

        if not filename:
            filename = os.path.join(
                self.data_dir,
                f'{self.datapath.split("/")[-1].split(".")[0]}_mordred.csv',
            )
        self.df_mordred.to_csv(filename, index=False)

        print(f"Saved Mordred descriptors to {filename}")

        return self.df_mordred

    def scale_train_data(self, mordred=False, raw_data=None):
        """
        Scale the training set of the original (DataWarrior descs) or Mordred data. Not
        needed for tree-based models (RF, XGBoost).

        Args:
            mordred (bool, optional): Whether to scale the Mordred data. Defaults to
            False.
            raw_data (str, optional): Path to the raw data file. If not provided,
            the default path will be used.
        """

        if raw_data:
            data = pd.read_csv(raw_data)
            scaler_path = raw_data.split(".")[0] + "_scaler.pkl"
        else:
            scaled_folder = os.path.join(self.data_dir, "scaled")
            basename = self.datapath.split("/")[-1].split(".")[0]
            scaler_path = os.path.join(scaled_folder, basename + "_scaler.pkl")
            raw_data = os.path.join(scaled_folder, basename + ".csv")
            if mordred:
                raw_data = raw_data.split(".")[0] + "_mordred.csv"
                assert self.df_mordred is not None, "Mordred data not loaded"
                data = self.df_mordred
            else:
                assert self.data is not None, "Data not loaded"
                data = self.data

        if mordred:
            scaler_path = scaler_path.split(".")[0] + "_mordred.pkl"
            assert set(MORDRED_DESCS).issubset(
                set(data.columns)
            ), "Data does not contain Mordred descriptors"
            assert len(set(MORDRED_DESCS).intersection(set(data.columns))) == len(
                MORDRED_DESCS
            ), "Data does not contain all Mordred descriptors"

            feat_cols = MORDRED_DESCS
            # array of features
            X = data[feat_cols].values
        else:
            assert set(FEATURES_DW).issubset(
                set(data.columns)
            ), "Data does not contain DataWarrior descriptors"
            assert (
                len(set(FEATURES_DW).intersection(set(data.columns))) == 10
            ), "Data does not contain all DataWarrior descriptors"
            feat_cols = FEATURES_DW
            # array of features
            X = data[feat_cols].values
        # array of targets
        y = data["target"].values

        y_scaled = y / 100

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # save scaler
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # save scaled data
        df_scaled = pd.DataFrame(X_scaled, columns=feat_cols)
        df_scaled["target"] = y_scaled
        df_scaled["SMILES"] = data["SMILES"]
        df_scaled.to_csv(raw_data.split(".")[0] + "_scaled.csv", index=False)

        print(f"Saved scaled data to {raw_data.split('.')[0] + '_scaled.csv'}")
        print(f"Saved scaler to {scaler_path}")

    def scale_test_data(
        self,
        mordred=False,
        raw_data=None,
    ):
        """
        Scale the test set of the original (DataWarrior descs) or Mordred data.
        Not needed for tree-based models (RF, XGBoost).

        Args:
            mordred (bool, optional): Whether to scale the Mordred data. Defaults to
            False.
            raw_data (str, optional): Path to the raw data file. If not provided,
            the default path will be used.
        """
        os.makedirs("data/scaled", exist_ok=True)

        if raw_data:
            data = pd.read_csv(raw_data)
            scaler_path = raw_data.split(".")[0] + "_scaler.pkl"
        else:
            scaler_path = self.datapath.split(".")[0] + "_scaler.pkl"
            raw_data = self.datapath.split(".")[0] + ".csv"
            if mordred:
                raw_data = raw_data.split(".")[0] + "_mordred.csv"
                assert self.df_mordred is not None, "Mordred data not loaded"
                data = self.df_mordred
            else:
                assert self.data is not None, "Data not loaded"
                data = self.data

        if mordred:
            scaler_path = scaler_path.split(".")[0] + "_mordred.pkl"
            assert set(MORDRED_DESCS).issubset(
                set(data.columns)
            ), "Data does not contain Mordred descriptors"
            assert len(set(MORDRED_DESCS).intersection(set(data.columns))) == len(
                MORDRED_DESCS
            ), "Data does not contain all Mordred descriptors"

            feat_cols = MORDRED_DESCS
            X = data[feat_cols].values
        else:
            assert set(FEATURES_DW).issubset(
                set(data.columns)
            ), "Data does not contain DataWarrior descriptors"
            assert (
                len(set(FEATURES_DW).intersection(set(data.columns))) == 10
            ), "Data does not contain all DataWarrior descriptors"
            feat_cols = FEATURES_DW
            # array of features
            X = data[feat_cols].values
        # array of targets
        y = data["target"].values

        y_scaled = y / 100

        # load scaler
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        X_scaled = scaler.transform(X)

        # save scaled data
        df_scaled = pd.DataFrame(X_scaled, columns=feat_cols)
        df_scaled["target"] = y_scaled
        df_scaled["SMILES"] = data["SMILES"]
        df_scaled.to_csv(raw_data.split(".")[0] + "_scaled.csv", index=False)

        print(f"Saved scaled data to {raw_data.split('.')[0] + '_scaled.csv'}")
        print(f"Saved scaler to {scaler_path}")
