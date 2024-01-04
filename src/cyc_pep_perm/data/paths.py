"""
Paths to data and models.
"""
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DATA_PATH = os.path.join(ROOT_PATH, "data")
TRAIN_RANDOM_DW = os.path.join(
    DATA_PATH, "Cyclic_peptide_membrane_permeability_random80percent.csv"
)
TRAIN_RANDOM_MORDRED = os.path.join(
    DATA_PATH, "Cyclic_peptide_membrane_permeability_random80percent_mordred.csv"
)

MODEL_PATH = os.path.join(ROOT_PATH, "models")
MODEL_RF_RANDOM_DW = os.path.join(MODEL_PATH, "rf_random_dw.pkl")
MODEL_RF_RANDOM_MORDRED = os.path.join(MODEL_PATH, "rf_random_mordred.pkl")
MODEL_XGB_RANDOM_DW = os.path.join(MODEL_PATH, "xgb_random_dw.pkl")
MODEL_XGB_RANDOM_MORDRED = os.path.join(MODEL_PATH, "xgb_random_mordred.pkl")
MODEL_RFCLASS_RANDOM_DW = os.path.join(MODEL_PATH, "rfclass_random_dw.pkl")
