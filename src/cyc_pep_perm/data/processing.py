import os

import pandas as pd
from pandas_ods_reader import read_ods
import mordred
from mordred import Calculator, descriptors
from rdkit import Chem


class DataProcessing:
    def __init__(
        self,
        datapath="data/raw/Cyclic_peptide_membrane_permeability_random80percent.ods",
        target_label="CAPA [1 µM]",
        smiles_label="SMILES",
    ):
        self.datapath = datapath
        assert os.path.exists(self.datapath), "File does not exist"
        self.target_label = target_label
        self.smiles_label = smiles_label
        self.data = None
        self.calculator = Calculator(descriptors, ignore_3D=True)

        self.read_data()

    def read_data(self):
        try:
            self.data = pd.read_csv(self.datapath)
        except Exception:
            self.data = read_ods(self.datapath, 1)

        self.smiles = self.data[self.smiles_label]
        self.mols = [Chem.MolFromSmiles(smile) for smile in self.smiles]

        self.data.rename(columns={self.target_label: "target"}, inplace=True)
        self.data.rename(columns={self.smiles_label: "SMILES"}, inplace=True)

    def calc_mordred(self, filename=None):
        df_mordred = self.calculator.pandas(self.mols)
        for col in df_mordred.columns:
            vals = df_mordred[col].values
            if all(isinstance(x, mordred.error.Error) for x in vals):
                df_mordred = df_mordred.drop(col, axis=1)

        df_mordred["SMILES"] = self.smiles
        df_mordred["target"] = self.data[self.target_label]

        assert len(df_mordred.columns) == 1599

        if not filename:
            filename = "data/raw/"\
                  f'{self.datapath.split("/")[-1].split(".")[0]}_mordred.csv'
        df_mordred.to_csv(filename, index=False)

    def scale_data(self):
        # TODO scale both original and mordred data
        os.path.makedirs("data/scaled", exist_ok=True)
        pass
