import numpy as np
import pandas as pd

from constants import QED_NAME, QED_THRESHOLD, NOISE_LEVELS
from utils import (
    compute_qed,
    append_flipped_labels,
    get_split_indexes,
    get_train_cs_path,
    get_test_cs_path,
)
from rdkit.Chem import MolFromSmiles

chids = ["CHEMBL3888429", "CHEMBL1909203", "CHEMBL1909140"]


def _save_train_test_smiles(df, chid):
    smiles_list = df.smiles.tolist()
    idx_train, idx_test = get_split_indexes(len(smiles_list))
    with open(get_train_cs_path(chid), "w") as f:
        f.writelines([smiles_list[ele] + "\n" for ele in idx_train])
    with open(get_test_cs_path(chid), "w") as f:
        f.writelines([smiles_list[ele] + "\n" for ele in idx_test])


def prepjak2(write=False):
    chid = "CHEMBL3888429"
    df = pd.read_csv(f"./assays/raw/{chid}.csv", sep=";")
    df = df[["Smiles", "pChEMBL Value"]]
    df.columns = ["smiles", "value"]

    label = np.array([1 if x > 8 else 0 for x in df.value])
    df["label"] = label
    df[QED_NAME] = df["smiles"].apply(
        lambda x: 1 * (compute_qed(MolFromSmiles(x)) >= QED_THRESHOLD)
    )
    df = append_flipped_labels(df, QED_NAME, NOISE_LEVELS)
    if write:
        df.to_csv(f"./assays/processed/{chid}.csv", index=None)
        _save_train_test_smiles(df, chid)
    return df


def prepegfr(write=False):
    chid = "CHEMBL1909203"
    df = pd.read_csv(f"./assays/raw/{chid}.csv", sep=";")
    df.head()

    # df[]
    df["label"] = pd.isna(df["Comment"]).astype("int")
    df["smiles"] = df["Smiles"]
    df = df[["smiles", "label"]]
    df = df.dropna()
    df[QED_NAME] = df["smiles"].apply(
        lambda x: 1 * (compute_qed(MolFromSmiles(x)) >= QED_THRESHOLD)
    )
    df = append_flipped_labels(df, QED_NAME, NOISE_LEVELS)
    if write:
        df.to_csv(f"./assays/processed/{chid}.csv", index=None)
        _save_train_test_smiles(df, chid)
    return df


def prepdrd2(write=False):
    chid = "CHEMBL1909140"
    df = pd.read_csv(f"./assays/raw/{chid}.csv", sep=";")
    df.head()

    # df[]
    df["label"] = pd.isna(df["Comment"]).astype("int")
    df["smiles"] = df["Smiles"]
    df = df[["smiles", "label"]]
    df = df.dropna()
    df[QED_NAME] = df["smiles"].apply(
        lambda x: 1 * (compute_qed(MolFromSmiles(x)) >= QED_THRESHOLD)
    )
    df = append_flipped_labels(df, QED_NAME, NOISE_LEVELS)
    df.to_csv(f"./assays/processed/{chid}.csv", index=None)
    _save_train_test_smiles(df, chid)

    return df


if __name__ == "__main__":
    prepegfr(write=True)
    prepjak2(write=True)
    prepdrd2(write=True)
