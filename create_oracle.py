import re

from rdkit import Chem
import pandas as pd
import numpy as np
import torch
from torch import nn
from rdkit.Chem import MACCSkeys
import random
import hashlib
import json
import sys
import os
from constants import COMPUTED_ORACLES, QED_NAME, SIMILARITY, QED_THRESHOLD
from utils import (
    compute_qed,
    ecfp,
    compute_similarity,
    get_df_from_chid,
    get_split_indexes,
)

FEATURE_SIZE = 167
BOTTLENECK_SIZE = 5


ORACLE_DIR = "oracles"
BLUEPRINT_DIR = os.path.join(ORACLE_DIR, "blueprints")
MODEL_DIRS = os.path.join(ORACLE_DIR, "models")
DATA_DIR = "assays/processed"


# typical architecture could be changed, here we choose for ex an
# exp non linearity on the last layer, because
# biological targets are often distributed on an exponential scale
class Network(nn.Module):
    def __init__(self, input_size, bottleneck_size, output_size, power):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, 1024, bias=False)
        self.l2 = nn.Linear(1024, 1024, bias=False)
        self.l3 = nn.Linear(1024, bottleneck_size, bias=False)
        self.l4 = nn.Linear(bottleneck_size, output_size, bias=False)

        # Pareto init for harder, more non linear models
        self.l1.state_dict()["weight"][:] = torch.Tensor(
            np.random.pareto(power, self.l1.state_dict()["weight"].size())
            - np.random.pareto(power, self.l1.state_dict()["weight"].size())
        )
        self.l2.state_dict()["weight"][:] = torch.Tensor(
            np.random.pareto(power, self.l2.state_dict()["weight"].size())
            - np.random.pareto(power, self.l2.state_dict()["weight"].size())
        )
        self.l3.state_dict()["weight"][:] = torch.Tensor(
            np.random.pareto(power, self.l3.state_dict()["weight"].size())
            - np.random.pareto(power, self.l3.state_dict()["weight"].size())
        )
        self.l4.state_dict()["weight"][:] = torch.Tensor(
            np.random.pareto(power, self.l4.state_dict()["weight"].size())
            - np.random.pareto(power, self.l4.state_dict()["weight"].size())
        )

    def forward(self, x):
        x = self.l1(x)
        x = torch.sigmoid(x)
        x = self.l2(x)
        x = torch.sigmoid(x)
        x = self.l3(x)
        x = torch.sin(x)
        x = self.l4(x)
        return torch.exp(x)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    dataset, seed, num_targets, power = args
    oracle_tag = _get_oracle_tag(dataset, num_targets, seed, power)
    if os.path.isdir(BLUEPRINT_DIR):
        list_blueprints = os.listdir(BLUEPRINT_DIR)
        full_bp_path = os.path.join(BLUEPRINT_DIR, f"blueprint_{oracle_tag}.json")
        if full_bp_path in list_blueprints:
            return (
                pd.read_csv(os.path.join(DATA_DIR, f"{dataset}.csv")),
                json.load(open(os.path.join(BLUEPRINT_DIR, full_bp_path))),
            )

    seed = int(seed)
    num_targets = int(num_targets)
    power = int(power)

    df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset}.csv"))
    mols = [Chem.MolFromSmiles(smi) for smi in list(df["smiles"])]
    maccs = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]

    # "true" inputs for the "truth" model
    input_fp = np.stack(maccs)

    input_size = input_fp.shape[1]

    output_size = num_targets  # number of simulated targets

    bottleneck_size = BOTTLENECK_SIZE  # size of penultimate layer : small value (~3)
    # create strong correlations between targets
    # bigger values (~100) create non correlated independent targets

    md5 = hashlib.md5((dataset + str(seed)).encode())
    seed_init = int(md5.hexdigest(), 16) % 2**25

    set_seed(seed_init)
    model = Network(input_size, bottleneck_size, output_size, power)

    synth_targets = model(torch.FloatTensor(input_fp))

    # constructing dataframe of generated synthetic targets
    df_gen = pd.DataFrame(synth_targets.detach().numpy())
    target_list = [
        _get_target_name(seed, num_targets, power, idx) for idx in df_gen.columns
    ]
    df_gen.columns = target_list

    # choosing thresholds for each targets
    # (checking that training set must not have solutions,
    # but solutions should exist in lead opt test set)
    thresholds_dict = _choose_threshold(df_gen, target_list)
    if not os.path.exists(ORACLE_DIR):
        os.makedirs(ORACLE_DIR)
    for target, threshold in thresholds_dict.items():
        df[target + "_continuous"] = df_gen[target]
        df[target] = 1 * (df_gen[target] >= threshold)
    # In the multi-objective case, remove solutions from training set which are in TPP
    if num_targets >= 2:
        df.loc[
            df[list(thresholds_dict.keys())].sum(1) == num_targets,
            list(thresholds_dict.keys()),
        ] = df[df[list(thresholds_dict.keys())].sum(1) == num_targets][
            list(thresholds_dict.keys())
        ].replace(
            1, np.nan
        )
    _save_oracle(dataset, model, num_targets, power, seed)
    _save_training_set(dataset, df)
    _save_blueprint(dataset, thresholds_dict, num_targets, power, seed)
    print("DONE: Oracle, train dataset and blueprint are saved")
    return df_gen, thresholds_dict


def _get_target_name(seed, num_targets, power, idx):
    return f"target_{num_targets}targs_power{power}_seed{seed}_targid{idx}"


def _choose_threshold(df_gen, target_list):
    return {target: np.percentile(df_gen[target].values, 50) for target in target_list}


def _get_oracle_tag(dataset, num_targets, seed, power):
    return (
        dataset
        + "_numtargets"
        + str(num_targets)
        + "_seed"
        + str(seed)
        + "_power"
        + str(power)
    )


def _save_blueprint(dataset, dict_threshold, num_targets, power, seed):
    if not os.path.exists(BLUEPRINT_DIR):
        os.makedirs(BLUEPRINT_DIR)
    json.dump(
        dict_threshold,
        open(
            os.path.join(
                BLUEPRINT_DIR,
                "blueprint"
                + _get_oracle_tag(dataset, num_targets, seed, power)
                + ".json",
            ),
            "w",
        ),
    )


def _save_training_set(dataset, df):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    df.to_csv(
        os.path.join(
            DATA_DIR,
            f"{dataset}.csv",
        ),
        index=False,
    )


def _save_oracle(dataset, model, num_targets, power, seed):
    if not os.path.exists(MODEL_DIRS):
        os.makedirs(MODEL_DIRS)
    torch.save(
        model.state_dict(),
        open(
            os.path.join(
                MODEL_DIRS,
                "oracle_" + _get_oracle_tag(dataset, num_targets, seed, power) + ".p",
            ),
            "wb",
        ),
    )


def _oracle_scoring(smiles_list, model):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    maccs = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
    scores = model(torch.FloatTensor(maccs))
    res = pd.DataFrame(scores.detach().numpy())
    return {idx: res[idx].tolist() for idx in res.columns}


def _extract_params_from_target_name(target_name):
    num_targets, power, seed, _ = re.findall("(\d+)", target_name)  # noqa
    return int(num_targets), int(power), int(seed)


def extract_oracles_from_target_names(dataset, target_names):
    oracles = []
    unique_params = []
    for target_name in target_names:
        if target_name in COMPUTED_ORACLES:
            oracles.append(get_computed_oracle(dataset, target_name))
            continue
        params = _extract_params_from_target_name(target_name)
        if params in unique_params:
            continue
        else:
            try:
                oracles.append(Oracle(dataset, *params))
                unique_params.append(params)
            except ValueError:
                print(f"No oracle found for {target_name}")
    return oracles


class Oracle:
    def __init__(self, dataset, num_targets, power, seed):
        self.model = load_oracle_model(dataset, num_targets, power, seed)
        self.target_lists = [
            _get_target_name(seed, num_targets, power, idx)
            for idx in range(num_targets)
        ]
        self.blueprint = load_blueprint(dataset, num_targets, power, seed)

    def __score_raw(self, smiles_list):
        scores = _oracle_scoring(smiles_list, self.model)
        return {self.target_lists[idx]: score for idx, score in scores.items()}

    def __bin_scores(self, scores):
        return {
            target_name: [val >= self.blueprint[target_name] for val in score]
            for target_name, score in scores.items()
        }

    def score(self, smiles_list):
        scores_raw = self.__score_raw(smiles_list)
        scores_binned = self.__bin_scores(scores_raw)
        return dict(
            **{f"oracle_raw_{target}": scores for target, scores in scores_raw.items()},
            **{
                f"oracle_binned_{target}": scores
                for target, scores in scores_binned.items()
            },
        )


class QEDScorer:
    def __init__(self, target_name):
        self.target_name = QED_NAME

    def score(self, smiles_list):
        scores_raw = self.raw_score_list(smiles_list)
        scores_binned = [1 * (ele > QED_THRESHOLD) for ele in scores_raw]
        return {
            self.target_name: scores_raw,
            self.target_name + "_binned": scores_binned,
        }

    def raw_score_list(self, smiles_list):
        return [compute_qed(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]


class SimilarityScorer:
    target_name = SIMILARITY

    def __init__(self, chid):
        df = get_df_from_chid(chid)
        train_idx, _ = get_split_indexes(len(df))
        self.train_fingerprints = ecfp(df.smiles.values[train_idx])
        self.similarity_max = os.getenv("similarity_max")
        if self.similarity_max:
            print(f"Similarity will not be optimized above {self.similarity_max}")
            self.similarity_max = float(self.similarity_max)

    def score(self, smiles_list):
        return {SIMILARITY: self.__raw_score_list(smiles_list)}

    def __raw_score_list(self, smiles_list):
        fp_list = ecfp(smiles_list)
        return [compute_similarity(fp, self.train_fingerprints) for fp in fp_list]

    def raw_score_list(self, smiles_list):
        fp_list = ecfp(smiles_list)
        if self.similarity_max:
            return [
                min(
                    compute_similarity(fp, self.train_fingerprints)
                    / self.similarity_max,
                    1,
                )
                for fp in fp_list
            ]
        return [compute_similarity(fp, self.train_fingerprints) for fp in fp_list]


def get_computed_oracle(chid, target_name):
    if QED_NAME in target_name:
        return QEDScorer(target_name)
    elif target_name == SIMILARITY:
        return SimilarityScorer(chid)
    else:
        raise ValueError(f"Not defined yet {target_name} for {chid}")


def load_oracle_model(dataset, num_targets, power, seed):
    model = Network(FEATURE_SIZE, BOTTLENECK_SIZE, int(num_targets), int(power))
    model_path = os.path.join(
        MODEL_DIRS,
        "oracle_" + _get_oracle_tag(dataset, num_targets, seed, power) + ".p",
    )
    if not os.path.exists(model_path):
        raise ValueError("Oracle model does not exist.")

    model.load_state_dict(torch.load(model_path))
    return model


def load_blueprint(dataset, num_targets, power, seed):
    blueprint_path = os.path.join(
        BLUEPRINT_DIR,
        "blueprint" + _get_oracle_tag(dataset, num_targets, seed, power) + ".json",
    )
    if not os.path.exists(blueprint_path):
        raise ValueError("Blueprint dict does not exist.")
    return json.load(open(blueprint_path, "r"))


if __name__ == "__main__":
    main(sys.argv[1:])
