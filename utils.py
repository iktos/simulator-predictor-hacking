import uuid
from functools import partial
from multiprocessing import Pool
from time import gmtime, strftime
from typing import List

import numpy as np
import pandas as pd
from guacamol.scoring_function import BatchScoringFunction
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Optional


def timestamp(adduuid=False):
    s = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    if adduuid:
        s = s + "_" + uuid.uuid4().hex
    return s


def get_train_cs_path(chid):
    return f"./assays/processed/{chid}_train.smiles"


def get_test_cs_path(chid):
    return f"./assays/processed/{chid}_test.smiles"


def get_split_indexes(n_sample):
    return train_test_split(np.arange(n_sample), test_size=0.5, random_state=19)


def can_list(smiles):
    ms = [Chem.MolFromSmiles(s) for s in smiles]
    return [Chem.MolToSmiles(m) for m in ms if m is not None]


def one_ecfp(smile, radius=2):
    "Calculate ECFP fingerprint. If smiles is invalid return none"
    try:
        m = Chem.MolFromSmiles(smile)
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=1024)
    except Exception as e:
        print(e)
        return None


def compute_qed(mol: str) -> Optional[float]:
    if mol is None:
        return None
    return Chem.QED.qed(mol)


def compute_similarity(fp, reference_fps):
    return max(BulkTanimotoSimilarity(fp, reference_fps))


def get_df_from_chid(chid):
    assay_file = f"./assays/processed/{chid}.csv"
    print(f"Reading data from: {assay_file}")
    df = pd.read_csv(assay_file)
    return df


def flip_labels(df: pd.DataFrame, column: str, noise_proportion: float) -> pd.DataFrame:
    res = df.copy()
    value_counts = res[column].value_counts()
    n_minority_class = value_counts.min()
    n_to_flip = int(noise_proportion * n_minority_class)
    for label in value_counts.index:
        res.loc[
            res[res[column] == label].sample(n=n_to_flip, replace=False).index, column
        ] = (1 - label)
    return res


def append_flipped_labels(
    df: pd.DataFrame, original_column: str, noise_percentages: List[int]
) -> pd.DataFrame:
    for noise in noise_percentages:
        noised_column = f"{original_column}_{noise}"
        df[noised_column] = df[original_column]
        df = flip_labels(df, noised_column, noise / 100)
    return df


def ecfp(smiles, radius=2, n_jobs=12):
    with Pool(n_jobs) as pool:
        X = pool.map(partial(one_ecfp, radius=radius), smiles)
    return X


def calc_auc(clf, X_test, y_test):
    preds = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)


def score(smiles_list, clf):
    """Makes predictions for a list of smiles. Returns none if smiles is invalid"""
    X = ecfp(smiles_list)
    X_valid = [x for x in X if x is not None]
    if len(X_valid) == 0:
        return X

    preds_valid = clf.predict_proba(np.array(X_valid))[:, 1]
    preds = []
    i = 0
    for li, x in enumerate(X):
        if x is None:
            # print(smiles_list[li], Chem.MolFromSmiles(smiles_list[li]), x)
            preds.append(None)
        else:
            preds.append(preds_valid[i])
            assert preds_valid[i] is not None
            i += 1
    return preds


class TPScoringFunction(BatchScoringFunction):
    def __init__(self, target_name, clf):
        super().__init__()
        self.target_name = target_name
        self.clf = clf

    def raw_score_list(self, smiles_list):
        return score(smiles_list, self.clf)


class MultiObjScoringFunction(BatchScoringFunction):
    def __init__(self, scoring_functions=List[BatchScoringFunction]):
        super().__init__()
        self._scoring_functions = scoring_functions

    def raw_score_list_detailed(self, smiles_list):
        return {
            scoring_fn.target_name: scoring_fn.raw_score_list(smiles_list)
            for scoring_fn in self._scoring_functions
        }

    def raw_score_list(self, smiles_list):
        return np.mean(
            [
                scoring_fn.raw_score_list(smiles_list)
                for scoring_fn in self._scoring_functions
            ],
            axis=0,
        )


class EnsembleScorer:
    def __init__(self, clfs: list):
        self.clfs = clfs

    def predict_proba(self, X: np.ndarray):
        return np.mean([clf.predict_proba(X) for clf in self.clfs], axis=0)
