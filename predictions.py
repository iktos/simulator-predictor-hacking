"""This script fits classifiers with different random seeds using
different splits of data. This can then be used to get a better estimate
of which optimization/control scores combinations are likely for training
and test data
"""
import json
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from create_oracle import get_computed_oracle
from utils import ecfp, get_df_from_chid
from constants import (
    MODELS_DIR,
    MODEL_NAME,
    MODEL_CONTROL,
    DATA_CONTROL,
    COMPUTED_TARGETS,
    RESCORER_NAME,
    RESCORER_DATA_ENSEMBLE_NAME,
    RESCORER_DATA_NAME,
)
from predictive_metrics import score_model


def get_model_providers(chid, target_names, model_type="rf"):
    assert model_type in [
        "rf",
        "lr",
    ], "Error: model_type should be either set to 'rf' or 'lr'"
    predictive_models = []
    computed_scores = []
    for target_name in target_names:
        if target_name in COMPUTED_TARGETS:
            computed_scores.append(ComputedValueProvider(chid, target_name))
        else:
            predictive_models.append(ModelProvider(chid, target_name, model_type))
    return predictive_models, computed_scores


class ComputedValueProvider:
    def __init__(self, chid, target_name):
        self.scorer = get_computed_oracle(chid, target_name)

    def get_model(self):
        return self.scorer


class ModelProvider:
    def __init__(self, chid, target_name, model_type):
        dir_results = os.path.join(MODELS_DIR, chid, target_name, model_type)
        if not os.path.exists(dir_results):
            os.makedirs(dir_results)
        self.model_path = os.path.join(dir_results, MODEL_NAME)
        self.rescorer_path = os.path.join(dir_results, RESCORER_NAME)
        self.rescorer_data_path = os.path.join(dir_results, RESCORER_DATA_NAME)
        self.rescorer_data_ensemble_path = os.path.join(
            dir_results, RESCORER_DATA_ENSEMBLE_NAME
        )
        self.model_control_path = os.path.join(dir_results, MODEL_CONTROL)
        self.data_control_path = os.path.join(dir_results, DATA_CONTROL)
        self.target_name = target_name
        self.model_type = model_type
        self.__run_training(chid, target_name)

    def __run_training(self, chid, target_name):
        if not os.path.exists(self.model_path + "_metrics.json"):
            print("MODEL DOES NOT EXIST, TRAINING")
            train_and_save_models(
                chid,
                target_name,
                self.model_path,
                self.model_control_path,
                self.data_control_path,
                self.model_type,
            )
            print("TRAINING DONE")

        if not os.path.exists(self.rescorer_path + "_metrics.json"):
            print("RESCORER MODEL DOES NOT EXIST, TRAINING")
            train_and_save_rescorer(
                chid,
                target_name,
                self.rescorer_path,
                self.rescorer_data_path,
                self.rescorer_data_ensemble_path,
                None,
                self.model_type,
            )
            print("TRAINING DONE")

    def get_model_optim(self):
        return pickle.load(open(self.model_path + ".p", "rb"))

    def get_model_control(self):
        return pickle.load(open(self.model_control_path + ".p", "rb"))

    def get_data_control(self):
        return pickle.load(open(self.data_control_path + ".p", "rb"))

    def get_rescorer(self):
        return pickle.load(open(self.rescorer_path + ".p", "rb"))

    def get_rescorer_data(self):
        return pickle.load(open(self.rescorer_data_path + ".p", "rb"))

    def get_rescorer_data_ensemble(self):
        return pickle.load(open(self.rescorer_data_ensemble_path + ".p", "rb"))

    def get_model_optim_metrics(self):
        return json.load(open(self.model_path + "_metrics.json", "r"))

    def get_model_control_metrics(self):
        return json.load(open(self.model_control_path + "_metrics.json", "r"))

    def get_data_control_metrics(self):
        return json.load(open(self.data_control_path + "_metrics.json", "r"))

    def get_rescorer_metrics(self):
        return json.load(open(self.rescorer_path + "_metrics.json", "r"))

    def get_rescorer_data_metrics(self):
        return json.load(open(self.rescorer_data_path + "_metrics.json", "r"))

    def get_rescorer_data_ensemble_metrics(self):
        return json.load(open(self.rescorer_data_ensemble_path + "_metrics.json", "r"))


def get_rf_model():
    return RandomForestClassifier(n_estimators=100, random_state=0)


def get_lr_model():
    return LogisticRegressionCV(
        solver="liblinear",
        Cs=np.logspace(np.exp(-4), 2, num=10, endpoint=True, base=np.exp(1)) - 1,
        scoring="roc_auc",
        cv=StratifiedKFold(4, shuffle=True, random_state=1),
        n_jobs=-1,
        penalty="l2",
    )


def train_and_save_models(
    chid,
    target_name: str,
    model_path,
    model_control_path,
    data_control_path,
    model_type,
):
    # read data and calculate ecfp fingerprints
    df = get_df_from_chid(chid)
    # removing NaNs values which correspond to the solutions in TPP in multiobjective cases
    df = df.dropna(subset=[target_name]).reset_index(drop=True)
    X = np.array(ecfp(df.smiles))
    y = np.array(df[target_name])

    # idx_train, idx_test = train_test_split(
    #     np.arange(len(y)), test_size=0.1, random_state=19
    # )

    # X1, X2, y1, y2 = X[idx_train], X[idx_test], y[idx_train], y[idx_test]

    dico_models = {"rf": get_rf_model, "lr": get_lr_model}

    model_optim = dico_models[model_type]()

    # Now the model optim is trained on the entire dataset
    model_optim.fit(X, y)
    predictions = model_optim.predict_proba(X)[:, 1]
    prediction_threshold = np.percentile(predictions, 100 * (1 - np.sum(y) / len(y)))

    model_optim_perfs = score_model(y, predictions, prediction_threshold)

    model_control = dico_models[model_type]()
    np.random.seed(10)
    # duplicating random point, so model_control could be adapted to LR model also
    idxes_to_keep = list(range(len(y)))
    idxes_to_keep = np.hstack(
        [idxes_to_keep, np.array(np.random.choice(idxes_to_keep))]
    )
    X1m = X[idxes_to_keep]
    y1m = y[idxes_to_keep]
    model_control.fit(X1m, y1m)
    predictions = model_control.predict_proba(X)[:, 1]
    prediction_threshold = np.percentile(predictions, 100 * (1 - np.sum(y) / len(y)))
    model_control_perfs = score_model(y, predictions, prediction_threshold)

    # data_control = dico_models[model_type]()
    # data_control.fit(X2, y2)
    # predictions = data_control.predict_proba(X1)[:, 1]
    # prediction_threshold = np.percentile(predictions, 100 * (1 - np.sum(y1) / len(y2)))
    # data_control_perfs = score_model(y1, predictions, prediction_threshold)

    for model, metrics, model_path in zip(
        [model_optim, model_control],
        [model_optim_perfs, model_control_perfs],
        [model_path, model_control_path],
    ):
        pickle.dump(model, open(model_path + ".p", "wb"))
        json.dump(metrics, open(model_path + "_metrics.json", "w"))


def train_and_save_rescorer(
    chid,
    target_name: str,
    rescorer_path,
    rescorer_data_path,
    rescorer_data_ensemble_path,
    data_control_model,
    model_type,
):
    # the rescorer is the opposite of the model optimized
    dico_rescorer_models = {"rf": get_lr_model, "lr": get_rf_model}

    df = get_df_from_chid(chid)
    X = np.array(ecfp(df.smiles))
    y = np.array(df[target_name])

    # idx_train, idx_test = get_split_indexes(len(y))
    #
    # X1, X2, y1, y2 = X[idx_train], X[idx_test], y[idx_train], y[idx_test]

    # rescorer
    rescorer = dico_rescorer_models[model_type]()
    rescorer.fit(X, y)
    predictions = rescorer.predict_proba(X)[:, 1]
    prediction_threshold = np.percentile(predictions, 100 * (1 - np.sum(y) / len(y)))

    rescorer_perfs = score_model(y, predictions, prediction_threshold)

    # data rescorer
    # data_rescorer = dico_rescorer_models[model_type]()
    # data_rescorer.fit(X2, y2)
    # predictions = data_rescorer.predict_proba(X1)[:, 1]
    # prediction_threshold = np.percentile(predictions, 100 * (1 - np.sum(y2) / len(y2)))
    #
    # data_rescorer_perfs = score_model(y1, predictions, prediction_threshold)

    # data ensemble  rescorer
    # rescorer_data_ensemble = EnsembleScorer([data_rescorer, data_control_model])
    # predictions = rescorer_data_ensemble.predict_proba(X1)[:, 1]
    # prediction_threshold = np.percentile(predictions, 100 * (1 - np.sum(y2) / len(y2)))
    #
    # rescorer_data_ensemble_perfs = score_model(y1, predictions, prediction_threshold)

    for model, metrics, model_path in zip(
        [rescorer],
        [rescorer_perfs],
        [rescorer_path],
    ):
        pickle.dump(model, open(model_path + ".p", "wb"))
        json.dump(metrics, open(model_path + "_metrics.json", "w"))
