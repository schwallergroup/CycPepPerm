"""
This file contains the descriptor names used in the project.

"""

FEATURES_DW = [
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

# filtered all 2D mordred descriptors (1613):
# 1) Remove all columns that lead to errors -> 1597
# 2) Remove all colunms with one single value -> 1200
# 3) Remove all collumns that have strings (mostly the EState descs) -> 1181
MORDRED_DESCS = [
    "SpAbs_A",
    "SpMax_A",
    "SpDiam_A",
    "SpAD_A",
    "SpMAD_A",
    "LogEE_A",
    "VE1_A",
    "VE2_A",
    "VE3_A",
    "VR1_A",
    "VR2_A",
    "VR3_A",
    "nAromAtom",
    "nAromBond",
    "nAtom",
    "nHeavyAtom",
    "nBridgehead",
    "nHetero",
    "nH",
    "nC",
    "nN",
    "nO",
    "ATS0dv",
    "ATS1dv",
    "ATS2dv",
    "ATS3dv",
    "ATS4dv",
    "ATS5dv",
    "ATS6dv",
    "ATS7dv",
    "ATS8dv",
    "ATS0d",
    "ATS1d",
    "ATS2d",
    "ATS3d",
    "ATS4d",
    "ATS5d",
    "ATS6d",
    "ATS7d",
    "ATS8d",
    "ATS0s",
    "ATS1s",
    "ATS2s",
    "ATS3s",
    "ATS4s",
    "ATS5s",
    "ATS6s",
    "ATS7s",
    "ATS8s",
    "ATS0Z",
    "ATS1Z",
    "ATS2Z",
    "ATS3Z",
    "ATS4Z",
    "ATS5Z",
    "ATS6Z",
    "ATS7Z",
    "ATS8Z",
    "ATS0m",
    "ATS1m",
    "ATS2m",
    "ATS3m",
    "ATS4m",
    "ATS5m",
    "ATS6m",
    "ATS7m",
    "ATS8m",
    "ATS0v",
    "ATS1v",
    "ATS2v",
    "ATS3v",
    "ATS4v",
    "ATS5v",
    "ATS6v",
    "ATS7v",
    "ATS8v",
    "ATS0se",
    "ATS1se",
    "ATS2se",
    "ATS3se",
    "ATS4se",
    "ATS5se",
    "ATS6se",
    "ATS7se",
    "ATS8se",
    "ATS0pe",
    "ATS1pe",
    "ATS2pe",
    "ATS3pe",
    "ATS4pe",
    "ATS5pe",
    "ATS6pe",
    "ATS7pe",
    "ATS8pe",
    "ATS0are",
    "ATS1are",
    "ATS2are",
    "ATS3are",
    "ATS4are",
    "ATS5are",
    "ATS6are",
    "ATS7are",
    "ATS8are",
    "ATS0p",
    "ATS1p",
    "ATS2p",
    "ATS3p",
    "ATS4p",
    "ATS5p",
    "ATS6p",
    "ATS7p",
    "ATS8p",
    "ATS0i",
    "ATS1i",
    "ATS2i",
    "ATS3i",
    "ATS4i",
    "ATS5i",
    "ATS6i",
    "ATS7i",
    "ATS8i",
    "AATS0dv",
    "AATS1dv",
    "AATS2dv",
    "AATS3dv",
    "AATS4dv",
    "AATS5dv",
    "AATS6dv",
    "AATS7dv",
    "AATS8dv",
    "AATS0d",
    "AATS1d",
    "AATS2d",
    "AATS3d",
    "AATS4d",
    "AATS5d",
    "AATS6d",
    "AATS7d",
    "AATS8d",
    "AATS0s",
    "AATS1s",
    "AATS2s",
    "AATS3s",
    "AATS4s",
    "AATS5s",
    "AATS6s",
    "AATS7s",
    "AATS8s",
    "AATS0Z",
    "AATS1Z",
    "AATS2Z",
    "AATS3Z",
    "AATS4Z",
    "AATS5Z",
    "AATS6Z",
    "AATS7Z",
    "AATS8Z",
    "AATS0m",
    "AATS1m",
    "AATS2m",
    "AATS3m",
    "AATS4m",
    "AATS5m",
    "AATS6m",
    "AATS7m",
    "AATS8m",
    "AATS0v",
    "AATS1v",
    "AATS2v",
    "AATS3v",
    "AATS4v",
    "AATS5v",
    "AATS6v",
    "AATS7v",
    "AATS8v",
    "AATS0se",
    "AATS1se",
    "AATS2se",
    "AATS3se",
    "AATS4se",
    "AATS5se",
    "AATS6se",
    "AATS7se",
    "AATS8se",
    "AATS0pe",
    "AATS1pe",
    "AATS2pe",
    "AATS3pe",
    "AATS4pe",
    "AATS5pe",
    "AATS6pe",
    "AATS7pe",
    "AATS8pe",
    "AATS0are",
    "AATS1are",
    "AATS2are",
    "AATS3are",
    "AATS4are",
    "AATS5are",
    "AATS6are",
    "AATS7are",
    "AATS8are",
    "AATS0p",
    "AATS1p",
    "AATS2p",
    "AATS3p",
    "AATS4p",
    "AATS5p",
    "AATS6p",
    "AATS7p",
    "AATS8p",
    "AATS0i",
    "AATS1i",
    "AATS2i",
    "AATS3i",
    "AATS4i",
    "AATS5i",
    "AATS6i",
    "AATS7i",
    "AATS8i",
    "ATSC0c",
    "ATSC1c",
    "ATSC2c",
    "ATSC3c",
    "ATSC4c",
    "ATSC5c",
    "ATSC6c",
    "ATSC7c",
    "ATSC8c",
    "ATSC0dv",
    "ATSC1dv",
    "ATSC2dv",
    "ATSC3dv",
    "ATSC4dv",
    "ATSC5dv",
    "ATSC6dv",
    "ATSC7dv",
    "ATSC8dv",
    "ATSC0d",
    "ATSC1d",
    "ATSC2d",
    "ATSC3d",
    "ATSC4d",
    "ATSC5d",
    "ATSC6d",
    "ATSC7d",
    "ATSC8d",
    "ATSC0s",
    "ATSC1s",
    "ATSC2s",
    "ATSC3s",
    "ATSC4s",
    "ATSC5s",
    "ATSC6s",
    "ATSC7s",
    "ATSC8s",
    "ATSC0Z",
    "ATSC1Z",
    "ATSC2Z",
    "ATSC3Z",
    "ATSC4Z",
    "ATSC5Z",
    "ATSC6Z",
    "ATSC7Z",
    "ATSC8Z",
    "ATSC0m",
    "ATSC1m",
    "ATSC2m",
    "ATSC3m",
    "ATSC4m",
    "ATSC5m",
    "ATSC6m",
    "ATSC7m",
    "ATSC8m",
    "ATSC0v",
    "ATSC1v",
    "ATSC2v",
    "ATSC3v",
    "ATSC4v",
    "ATSC5v",
    "ATSC6v",
    "ATSC7v",
    "ATSC8v",
    "ATSC0se",
    "ATSC1se",
    "ATSC2se",
    "ATSC3se",
    "ATSC4se",
    "ATSC5se",
    "ATSC6se",
    "ATSC7se",
    "ATSC8se",
    "ATSC0pe",
    "ATSC1pe",
    "ATSC2pe",
    "ATSC3pe",
    "ATSC4pe",
    "ATSC5pe",
    "ATSC6pe",
    "ATSC7pe",
    "ATSC8pe",
    "ATSC0are",
    "ATSC1are",
    "ATSC2are",
    "ATSC3are",
    "ATSC4are",
    "ATSC5are",
    "ATSC6are",
    "ATSC7are",
    "ATSC8are",
    "ATSC0p",
    "ATSC1p",
    "ATSC2p",
    "ATSC3p",
    "ATSC4p",
    "ATSC5p",
    "ATSC6p",
    "ATSC7p",
    "ATSC8p",
    "ATSC0i",
    "ATSC1i",
    "ATSC2i",
    "ATSC3i",
    "ATSC4i",
    "ATSC5i",
    "ATSC6i",
    "ATSC7i",
    "ATSC8i",
    "AATSC0c",
    "AATSC1c",
    "AATSC2c",
    "AATSC3c",
    "AATSC4c",
    "AATSC5c",
    "AATSC6c",
    "AATSC7c",
    "AATSC8c",
    "AATSC0dv",
    "AATSC1dv",
    "AATSC2dv",
    "AATSC3dv",
    "AATSC4dv",
    "AATSC5dv",
    "AATSC6dv",
    "AATSC7dv",
    "AATSC8dv",
    "AATSC0d",
    "AATSC1d",
    "AATSC2d",
    "AATSC3d",
    "AATSC4d",
    "AATSC5d",
    "AATSC6d",
    "AATSC7d",
    "AATSC8d",
    "AATSC0s",
    "AATSC1s",
    "AATSC2s",
    "AATSC3s",
    "AATSC4s",
    "AATSC5s",
    "AATSC6s",
    "AATSC7s",
    "AATSC8s",
    "AATSC0Z",
    "AATSC1Z",
    "AATSC2Z",
    "AATSC3Z",
    "AATSC4Z",
    "AATSC5Z",
    "AATSC6Z",
    "AATSC7Z",
    "AATSC8Z",
    "AATSC0m",
    "AATSC1m",
    "AATSC2m",
    "AATSC3m",
    "AATSC4m",
    "AATSC5m",
    "AATSC6m",
    "AATSC7m",
    "AATSC8m",
    "AATSC0v",
    "AATSC1v",
    "AATSC2v",
    "AATSC3v",
    "AATSC4v",
    "AATSC5v",
    "AATSC6v",
    "AATSC7v",
    "AATSC8v",
    "AATSC0se",
    "AATSC1se",
    "AATSC2se",
    "AATSC3se",
    "AATSC4se",
    "AATSC5se",
    "AATSC6se",
    "AATSC7se",
    "AATSC8se",
    "AATSC0pe",
    "AATSC1pe",
    "AATSC2pe",
    "AATSC3pe",
    "AATSC4pe",
    "AATSC5pe",
    "AATSC6pe",
    "AATSC7pe",
    "AATSC8pe",
    "AATSC0are",
    "AATSC1are",
    "AATSC2are",
    "AATSC3are",
    "AATSC4are",
    "AATSC5are",
    "AATSC6are",
    "AATSC7are",
    "AATSC8are",
    "AATSC0p",
    "AATSC1p",
    "AATSC2p",
    "AATSC3p",
    "AATSC4p",
    "AATSC5p",
    "AATSC6p",
    "AATSC7p",
    "AATSC8p",
    "AATSC0i",
    "AATSC1i",
    "AATSC2i",
    "AATSC3i",
    "AATSC4i",
    "AATSC5i",
    "AATSC6i",
    "AATSC7i",
    "AATSC8i",
    "MATS1c",
    "MATS2c",
    "MATS3c",
    "MATS4c",
    "MATS5c",
    "MATS6c",
    "MATS7c",
    "MATS8c",
    "MATS1dv",
    "MATS2dv",
    "MATS3dv",
    "MATS4dv",
    "MATS5dv",
    "MATS6dv",
    "MATS7dv",
    "MATS8dv",
    "MATS1d",
    "MATS2d",
    "MATS3d",
    "MATS4d",
    "MATS5d",
    "MATS6d",
    "MATS7d",
    "MATS8d",
    "MATS1s",
    "MATS2s",
    "MATS3s",
    "MATS4s",
    "MATS5s",
    "MATS6s",
    "MATS7s",
    "MATS8s",
    "MATS1Z",
    "MATS2Z",
    "MATS3Z",
    "MATS4Z",
    "MATS5Z",
    "MATS6Z",
    "MATS7Z",
    "MATS8Z",
    "MATS1m",
    "MATS2m",
    "MATS3m",
    "MATS4m",
    "MATS5m",
    "MATS6m",
    "MATS7m",
    "MATS8m",
    "MATS1v",
    "MATS2v",
    "MATS3v",
    "MATS4v",
    "MATS5v",
    "MATS6v",
    "MATS7v",
    "MATS8v",
    "MATS1se",
    "MATS2se",
    "MATS3se",
    "MATS4se",
    "MATS5se",
    "MATS6se",
    "MATS7se",
    "MATS8se",
    "MATS1pe",
    "MATS2pe",
    "MATS3pe",
    "MATS4pe",
    "MATS5pe",
    "MATS6pe",
    "MATS7pe",
    "MATS8pe",
    "MATS1are",
    "MATS2are",
    "MATS3are",
    "MATS4are",
    "MATS5are",
    "MATS6are",
    "MATS7are",
    "MATS8are",
    "MATS1p",
    "MATS2p",
    "MATS3p",
    "MATS4p",
    "MATS5p",
    "MATS6p",
    "MATS7p",
    "MATS8p",
    "MATS1i",
    "MATS2i",
    "MATS3i",
    "MATS4i",
    "MATS5i",
    "MATS6i",
    "MATS7i",
    "MATS8i",
    "GATS1c",
    "GATS2c",
    "GATS3c",
    "GATS4c",
    "GATS5c",
    "GATS6c",
    "GATS7c",
    "GATS8c",
    "GATS1dv",
    "GATS2dv",
    "GATS3dv",
    "GATS4dv",
    "GATS5dv",
    "GATS6dv",
    "GATS7dv",
    "GATS8dv",
    "GATS1d",
    "GATS2d",
    "GATS3d",
    "GATS4d",
    "GATS5d",
    "GATS6d",
    "GATS7d",
    "GATS8d",
    "GATS1s",
    "GATS2s",
    "GATS3s",
    "GATS4s",
    "GATS5s",
    "GATS6s",
    "GATS7s",
    "GATS8s",
    "GATS1Z",
    "GATS2Z",
    "GATS3Z",
    "GATS4Z",
    "GATS5Z",
    "GATS6Z",
    "GATS7Z",
    "GATS8Z",
    "GATS1m",
    "GATS2m",
    "GATS3m",
    "GATS4m",
    "GATS5m",
    "GATS6m",
    "GATS7m",
    "GATS8m",
    "GATS1v",
    "GATS2v",
    "GATS3v",
    "GATS4v",
    "GATS5v",
    "GATS6v",
    "GATS7v",
    "GATS8v",
    "GATS1se",
    "GATS2se",
    "GATS3se",
    "GATS4se",
    "GATS5se",
    "GATS6se",
    "GATS7se",
    "GATS8se",
    "GATS1pe",
    "GATS2pe",
    "GATS3pe",
    "GATS4pe",
    "GATS5pe",
    "GATS6pe",
    "GATS7pe",
    "GATS8pe",
    "GATS1are",
    "GATS2are",
    "GATS3are",
    "GATS4are",
    "GATS5are",
    "GATS6are",
    "GATS7are",
    "GATS8are",
    "GATS1p",
    "GATS2p",
    "GATS3p",
    "GATS4p",
    "GATS5p",
    "GATS6p",
    "GATS7p",
    "GATS8p",
    "GATS1i",
    "GATS2i",
    "GATS3i",
    "GATS4i",
    "GATS5i",
    "GATS6i",
    "GATS7i",
    "GATS8i",
    "BCUTc-1h",
    "BCUTc-1l",
    "BCUTdv-1h",
    "BCUTdv-1l",
    "BCUTd-1h",
    "BCUTd-1l",
    "BCUTs-1h",
    "BCUTs-1l",
    "BCUTZ-1h",
    "BCUTZ-1l",
    "BCUTm-1h",
    "BCUTm-1l",
    "BCUTv-1h",
    "BCUTv-1l",
    "BCUTse-1h",
    "BCUTse-1l",
    "BCUTpe-1h",
    "BCUTpe-1l",
    "BCUTare-1h",
    "BCUTare-1l",
    "BCUTp-1h",
    "BCUTp-1l",
    "BCUTi-1h",
    "BCUTi-1l",
    "BalabanJ",
    "SpAbs_DzZ",
    "SpMax_DzZ",
    "SpDiam_DzZ",
    "SpAD_DzZ",
    "SpMAD_DzZ",
    "LogEE_DzZ",
    "SM1_DzZ",
    "VE1_DzZ",
    "VE2_DzZ",
    "VE3_DzZ",
    "VR1_DzZ",
    "VR2_DzZ",
    "VR3_DzZ",
    "SpAbs_Dzm",
    "SpMax_Dzm",
    "SpDiam_Dzm",
    "SpAD_Dzm",
    "SpMAD_Dzm",
    "LogEE_Dzm",
    "SM1_Dzm",
    "VE1_Dzm",
    "VE2_Dzm",
    "VE3_Dzm",
    "VR1_Dzm",
    "VR2_Dzm",
    "VR3_Dzm",
    "SpAbs_Dzv",
    "SpMax_Dzv",
    "SpDiam_Dzv",
    "SpAD_Dzv",
    "SpMAD_Dzv",
    "LogEE_Dzv",
    "SM1_Dzv",
    "VE1_Dzv",
    "VE2_Dzv",
    "VE3_Dzv",
    "VR1_Dzv",
    "VR2_Dzv",
    "VR3_Dzv",
    "SpAbs_Dzse",
    "SpMax_Dzse",
    "SpDiam_Dzse",
    "SpAD_Dzse",
    "SpMAD_Dzse",
    "LogEE_Dzse",
    "SM1_Dzse",
    "VE1_Dzse",
    "VE2_Dzse",
    "VE3_Dzse",
    "VR1_Dzse",
    "VR2_Dzse",
    "VR3_Dzse",
    "SpAbs_Dzpe",
    "SpMax_Dzpe",
    "SpDiam_Dzpe",
    "SpAD_Dzpe",
    "SpMAD_Dzpe",
    "LogEE_Dzpe",
    "SM1_Dzpe",
    "VE1_Dzpe",
    "VE2_Dzpe",
    "VE3_Dzpe",
    "VR1_Dzpe",
    "VR2_Dzpe",
    "VR3_Dzpe",
    "SpAbs_Dzare",
    "SpMax_Dzare",
    "SpDiam_Dzare",
    "SpAD_Dzare",
    "SpMAD_Dzare",
    "LogEE_Dzare",
    "SM1_Dzare",
    "VE1_Dzare",
    "VE2_Dzare",
    "VE3_Dzare",
    "VR1_Dzare",
    "VR2_Dzare",
    "VR3_Dzare",
    "SpAbs_Dzp",
    "SpMax_Dzp",
    "SpDiam_Dzp",
    "SpAD_Dzp",
    "SpMAD_Dzp",
    "LogEE_Dzp",
    "SM1_Dzp",
    "VE1_Dzp",
    "VE2_Dzp",
    "VE3_Dzp",
    "VR1_Dzp",
    "VR2_Dzp",
    "VR3_Dzp",
    "SpAbs_Dzi",
    "SpMax_Dzi",
    "SpDiam_Dzi",
    "SpAD_Dzi",
    "SpMAD_Dzi",
    "LogEE_Dzi",
    "SM1_Dzi",
    "VE1_Dzi",
    "VE2_Dzi",
    "VE3_Dzi",
    "VR1_Dzi",
    "VR2_Dzi",
    "VR3_Dzi",
    "BertzCT",
    "nBonds",
    "nBondsO",
    "nBondsS",
    "nBondsD",
    "nBondsA",
    "nBondsM",
    "nBondsKS",
    "nBondsKD",
    "RNCG",
    "RPCG",
    "C1SP2",
    "C2SP2",
    "C3SP2",
    "C1SP3",
    "C2SP3",
    "C3SP3",
    "HybRatio",
    "FCSP3",
    "Xch-4d",
    "Xch-5d",
    "Xch-6d",
    "Xch-7d",
    "Xch-4dv",
    "Xch-5dv",
    "Xch-6dv",
    "Xch-7dv",
    "Xc-3d",
    "Xc-5d",
    "Xc-3dv",
    "Xc-5dv",
    "Xpc-4d",
    "Xpc-5d",
    "Xpc-6d",
    "Xpc-4dv",
    "Xpc-5dv",
    "Xpc-6dv",
    "Xp-0d",
    "Xp-1d",
    "Xp-2d",
    "Xp-3d",
    "Xp-4d",
    "Xp-5d",
    "Xp-6d",
    "Xp-7d",
    "AXp-0d",
    "AXp-1d",
    "AXp-2d",
    "AXp-3d",
    "AXp-4d",
    "AXp-5d",
    "AXp-6d",
    "AXp-7d",
    "Xp-0dv",
    "Xp-1dv",
    "Xp-2dv",
    "Xp-3dv",
    "Xp-4dv",
    "Xp-5dv",
    "Xp-6dv",
    "Xp-7dv",
    "AXp-0dv",
    "AXp-1dv",
    "AXp-2dv",
    "AXp-3dv",
    "AXp-4dv",
    "AXp-5dv",
    "AXp-6dv",
    "AXp-7dv",
    "SZ",
    "Sm",
    "Sv",
    "Sse",
    "Spe",
    "Sare",
    "Sp",
    "Si",
    "MZ",
    "Mm",
    "Mv",
    "Mse",
    "Mpe",
    "Mare",
    "Mp",
    "Mi",
    "SpAbs_D",
    "SpMax_D",
    "SpDiam_D",
    "SpAD_D",
    "SpMAD_D",
    "LogEE_D",
    "VE1_D",
    "VE2_D",
    "VE3_D",
    "VR1_D",
    "VR2_D",
    "VR3_D",
    "NsCH3",
    "NssCH2",
    "NaaCH",
    "NsssCH",
    "NdssC",
    "NaasC",
    "NaaaC",
    "NsNH2",
    "NssNH",
    "NaaNH",
    "NaaN",
    "NsssN",
    "NsOH",
    "NdO",
    "SsCH3",
    "SssCH2",
    "SaaCH",
    "SsssCH",
    "SdssC",
    "SaasC",
    "SaaaC",
    "SsNH2",
    "SssNH",
    "SaaNH",
    "SaaN",
    "SsssN",
    "SsOH",
    "SdO",
    "SssO",
    "SssS",
    "SsCl",
    "MAXssCH2",
    "MAXaaCH",
    "MAXsssCH",
    "MAXdssC",
    "MAXaasC",
    "MAXssNH",
    "MAXaaN",
    "MAXdO",
    "MAXssO",
    "MAXssS",
    "MAXsCl",
    "MINssCH2",
    "MINaaCH",
    "MINsssCH",
    "MINdssC",
    "MINaasC",
    "MINssNH",
    "MINaaN",
    "MINdO",
    "MINssO",
    "MINssS",
    "MINsCl",
    "ECIndex",
    "ETA_alpha",
    "AETA_alpha",
    "ETA_shape_p",
    "ETA_shape_y",
    "ETA_beta",
    "AETA_beta",
    "ETA_beta_s",
    "AETA_beta_s",
    "ETA_beta_ns",
    "AETA_beta_ns",
    "ETA_beta_ns_d",
    "AETA_beta_ns_d",
    "ETA_eta",
    "AETA_eta",
    "ETA_eta_L",
    "AETA_eta_L",
    "ETA_eta_R",
    "AETA_eta_R",
    "ETA_eta_RL",
    "AETA_eta_RL",
    "ETA_eta_F",
    "AETA_eta_F",
    "ETA_eta_FL",
    "AETA_eta_FL",
    "ETA_eta_B",
    "AETA_eta_B",
    "ETA_eta_BR",
    "AETA_eta_BR",
    "ETA_dAlpha_B",
    "ETA_epsilon_1",
    "ETA_epsilon_2",
    "ETA_epsilon_3",
    "ETA_epsilon_4",
    "ETA_epsilon_5",
    "ETA_dEpsilon_A",
    "ETA_dEpsilon_B",
    "ETA_dEpsilon_C",
    "ETA_dEpsilon_D",
    "ETA_dBeta",
    "AETA_dBeta",
    "ETA_psi_1",
    "ETA_dPsi_A",
    "fragCpx",
    "fMF",
    "nHBAcc",
    "nHBDon",
    "IC0",
    "IC1",
    "IC2",
    "IC3",
    "IC4",
    "IC5",
    "TIC0",
    "TIC1",
    "TIC2",
    "TIC3",
    "TIC4",
    "TIC5",
    "SIC0",
    "SIC1",
    "SIC2",
    "SIC3",
    "SIC4",
    "SIC5",
    "BIC0",
    "BIC1",
    "BIC2",
    "BIC3",
    "BIC4",
    "BIC5",
    "CIC0",
    "CIC1",
    "CIC2",
    "CIC3",
    "CIC4",
    "CIC5",
    "MIC0",
    "MIC1",
    "MIC2",
    "MIC3",
    "MIC4",
    "MIC5",
    "ZMIC0",
    "ZMIC1",
    "ZMIC2",
    "ZMIC3",
    "ZMIC4",
    "ZMIC5",
    "Kier1",
    "Kier2",
    "Kier3",
    "FilterItLogS",
    "VMcGowan",
    "LabuteASA",
    "PEOE_VSA1",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "PEOE_VSA10",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "SMR_VSA1",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA8",
    "SlogP_VSA11",
    "EState_VSA1",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "EState_VSA10",
    "VSA_EState1",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "MDEC-22",
    "MDEC-23",
    "MDEC-33",
    "MDEO-11",
    "MDEO-12",
    "MDEN-22",
    "MID",
    "AMID",
    "MID_h",
    "AMID_h",
    "MID_C",
    "AMID_C",
    "MID_N",
    "AMID_N",
    "MID_O",
    "AMID_O",
    "MID_X",
    "AMID_X",
    "MPC2",
    "MPC3",
    "MPC4",
    "MPC5",
    "MPC6",
    "MPC7",
    "MPC8",
    "MPC9",
    "MPC10",
    "TMPC10",
    "piPC1",
    "piPC2",
    "piPC3",
    "piPC4",
    "piPC5",
    "piPC6",
    "piPC7",
    "piPC8",
    "piPC9",
    "piPC10",
    "TpiPC10",
    "apol",
    "bpol",
    "nRing",
    "n4Ring",
    "n5Ring",
    "n6Ring",
    "nG12Ring",
    "nHRing",
    "n4HRing",
    "n5HRing",
    "n6HRing",
    "nG12HRing",
    "naRing",
    "n5aRing",
    "n6aRing",
    "naHRing",
    "n5aHRing",
    "nARing",
    "n4ARing",
    "n5ARing",
    "n6ARing",
    "nG12ARing",
    "nAHRing",
    "n4AHRing",
    "n5AHRing",
    "n6AHRing",
    "nG12AHRing",
    "nFRing",
    "n9FRing",
    "nFHRing",
    "n9FHRing",
    "nFaRing",
    "n9FaRing",
    "nFaHRing",
    "n9FaHRing",
    "nRot",
    "RotRatio",
    "SLogP",
    "SMR",
    "TopoPSA(NO)",
    "TopoPSA",
    "GGI1",
    "GGI2",
    "GGI3",
    "GGI4",
    "GGI5",
    "GGI6",
    "GGI7",
    "GGI8",
    "GGI9",
    "GGI10",
    "JGI1",
    "JGI2",
    "JGI3",
    "JGI4",
    "JGI5",
    "JGI6",
    "JGI7",
    "JGI8",
    "JGI9",
    "JGI10",
    "JGT10",
    "Diameter",
    "Radius",
    "TopoShapeIndex",
    "PetitjeanIndex",
    "Vabc",
    "VAdjMat",
    "MWC01",
    "MWC02",
    "MWC03",
    "MWC04",
    "MWC05",
    "MWC06",
    "MWC07",
    "MWC08",
    "MWC09",
    "MWC10",
    "TMWC10",
    "SRW02",
    "SRW04",
    "SRW05",
    "SRW06",
    "SRW07",
    "SRW08",
    "SRW09",
    "SRW10",
    "TSRW10",
    "MW",
    "AMW",
    "WPath",
    "WPol",
    "Zagreb1",
    "Zagreb2",
    "mZagreb1",
    "mZagreb2",
]
