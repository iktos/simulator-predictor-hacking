import os
import json
import numpy as np
import torch
from time import time
from optimizers import OPTIMIZER_FACTORY
from utils import timestamp, get_train_cs_path, get_df_from_chid, get_split_indexes
from scores import scoring_factory
from create_oracle import extract_oracles_from_target_names
from constants import opt_args, MODEL_NAME, SEP, DEFAULT_RESCORING


def optimize(  # noqa
    steps,
    seed,
    optimizer_name,
    optimizer_kwargs,
    chid,
    target_names,
    model_type,
    log_base,
):

    """
    Args:
        - seed: which random seed to use
        - optimizer_name: which optimizer to use (graph_ga or lstm_hc)
        - optimizer_kwargs: dictionary with arguments for the optimizer
        - scorer
        - log_base: Where to store results. Will be appended by timestamp
    """
    if optimizer_name == "graph_ga":
        optimizer_kwargs["generations"] = steps
    elif optimizer_name == "lstm_hc":
        optimizer_kwargs["n_epochs"] = steps
    else:
        raise ValueError("optimizer_name should be one of: graph_ga, lstm_hc")

    config = locals()
    print(chid, target_names)
    scorer_dict = scoring_factory(chid, model_type, target_names)

    optimizer = OPTIMIZER_FACTORY[optimizer_name](**optimizer_kwargs)
    oracles = extract_oracles_from_target_names(
        chid, list(set(target_names + DEFAULT_RESCORING))
    )

    if oracles:
        print("Retrieved an oracle & will rescore the ouput with it.")

    # Results might not be fully reproducible when using pytorch
    # https://pytorch.org/docs/stable/notes/randomness.html
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # set up logging
    results_dir = os.path.join(log_base, optimizer_name, timestamp())
    os.makedirs(results_dir)

    config_file = os.path.join(results_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(config, f)

    # use only molecules from train set for initialization (no molecule in TPP in multiobj)
    df = get_df_from_chid(chid)
    df = df.dropna(subset=[target_names[0]]).reset_index(drop=True)
    X = np.array(df.smiles)
    idx_train, idx_test = get_split_indexes(len(X))
    X1 = X[idx_train]

    t0 = time()
    print("Optimizing")
    smiles_history = optimizer.generate_optimized_molecules(
        scorer_dict[MODEL_NAME], 5, list(X1), get_history=True
    )
    t1 = time()
    opt_time = t1 - t0
    print(f"Optimization time {opt_time:.2f}")

    results = {"smiles": smiles_history}
    for scorer_name, scorer in scorer_dict.items():
        print(scorer.raw_score_list_detailed(["CNCNCC", "CCC", "COCC"]))
    for scorer_name, scorer in scorer_dict.items():
        results[scorer_name] = [
            scorer.raw_score_list_detailed(smiles_list)
            for smiles_list in smiles_history
        ]

    if oracles:
        for oracle in oracles:
            res_oracles = []
            for smiles_list in smiles_history:
                res_oracles.append(oracle.score(smiles_list))
            for target_name in res_oracles[0].keys():
                results[target_name] = [
                    {target_name: res[target_name]} for res in res_oracles
                ]

    # make a list of dictionaries for every time step
    # this is far from an optimal data structure

    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f)

    rescoring_time = time() - t1
    print(f"Rescoring time {rescoring_time:.2f}")

    print(f"Storing results in {results_dir}")


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--chid", type=str)
    parser.add_argument("--target_names", "--list", nargs="+")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--optimizer", type=str, help="Type of optimizer, should be available"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--use_train_cs", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)

    args = parser.parse_args()
    print("CHID ", args.chid)
    print("target_names ", args.target_names)
    print("use_train_cs ", args.use_train_cs)
    print("model_type ", args.model_type)

    if args.optimizer not in opt_args.keys():
        raise ValueError(f"generator argument should be in {list(opt_args.keys())}")

    optimizer_kwargs = opt_args[args.optimizer]
    if args.use_train_cs:
        optimizer_kwargs["smi_file"] = get_train_cs_path(args.chid)
    log_base = os.path.join(
        args.results_dir,
        args.chid + (args.use_train_cs == 1) * "_cs",
        SEP.join(args.target_names),
        args.model_type,
    )
    optimize(
        args.steps,
        args.seed,
        args.optimizer,
        optimizer_kwargs,
        args.chid,
        args.target_names,
        args.model_type,
        log_base,
    )
