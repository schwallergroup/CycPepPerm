"""
Paths to data and models.
"""

import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DATA_PATH = os.path.join(ROOT_PATH, "data")
TRAIN_RANDOM_DW = os.path.join(DATA_PATH, "perm_random80_train_dw.csv")
TRAIN_RANDOM_MORDRED = os.path.join(DATA_PATH, "perm_random80_train_mordred.csv")
TEST_RANDOM_DW = os.path.join(DATA_PATH, "perm_random20_test_dw.csv")
TEST_RANDOM_MORDRED = os.path.join(DATA_PATH, "perm_random20_test_mordred.csv")

MODEL_PATH = os.path.join(ROOT_PATH, "models")
MODEL_RF_RANDOM_DW = os.path.join(MODEL_PATH, "rf_random_dw.pkl")
MODEL_RF_RANDOM_MORDRED = os.path.join(MODEL_PATH, "rf_random_mordred.pkl")
MODEL_XGB_RANDOM_DW = os.path.join(MODEL_PATH, "xgb_random_dw.pkl")
MODEL_XGB_RANDOM_MORDRED = os.path.join(MODEL_PATH, "xgb_random_mordred.pkl")
MODEL_RFCLASS_RANDOM_DW = os.path.join(MODEL_PATH, "rfclass_random_dw.pkl")
