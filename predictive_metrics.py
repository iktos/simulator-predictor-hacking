import numpy as np
from typing import Tuple

from sklearn.metrics import (
    auc,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    cohen_kappa_score,
    accuracy_score,
    log_loss,
    brier_score_loss,
    f1_score,
)

np.seterr(divide="ignore", invalid="ignore")

FUNCTION = "function"
ENRICHMENT = "normalized precision"
KWARGS = "kwargs"
TYPE = "type"
SCORE = "score"
BIN = "bin"
BOTH = "both"


def proportion_active(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Computes the proportion of active in the true values
    :param y_true:
    :param y_predicted:
    :return:
    """
    return np.sum(y_true) / len(y_true)


def pr_auc_score(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Computes the area under the curve of the Precision-Recall curve for a
    classification task
    :param y_true: binary true values
    :param y_predicted: binary predicted values
    :return:
    """
    precision, recall, _ = precision_recall_curve(y_true, y_predicted)
    return auc(recall, precision)


def uncertainty_auc_score(
    y_true: np.ndarray, y_predicted: np.ndarray, score_predicted: np.ndarray
) -> float:
    """
    Computes the AUC-ROC using an uncertainty score to predict whether a sample is
    mispredicted. The uncertainty score here is 2 * (1 - max(P(y = c_i))) with c_i being
    the class i (for binary classification here, i=0 or 1), score is in [0, 1]
    ref: https://arxiv.org/abs/1811.02633
    :param y_true: binary true values
    :param y_predicted: binary predicted values
    :param score_predicted: probability of the binary prediction
    :return:
    """
    uncertainty = 2 * (
        1 - np.max(np.stack((score_predicted, 1 - score_predicted)), axis=0)
    )
    misprediction = (y_true != y_predicted).astype(np.int)
    if np.alltrue(misprediction == 1):
        auc = 0
    elif np.alltrue(misprediction == 0):
        auc = 1
    else:
        auc = roc_auc_score(misprediction, uncertainty)
    return auc


def rmse(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Computes the root mean squared error for a regression task
    :param y_true:
    :param y_predicted:
    :return:
    """
    return np.sqrt(mean_squared_error(y_true, y_predicted))


def _calibration_bins(
    y_true: np.ndarray, y_predicted: np.ndarray, n_bins=10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the calibration errors per bin, after binning in n_bins equal-length bins.
    Calibration error stands for the absolute value of the difference between the true
    fraction of positive instances in the bin with the mean of the output probabilities
    for the instances in the bins. We store the size of each bins for a weighted
    average.
    :param y_true:
    :param y_predicted:
    :param n_bins:
    :return:
    """
    mean_probabilities = np.zeros(n_bins, dtype=float)
    fraction_of_positives = np.zeros(n_bins, dtype=float)
    bins_size = np.zeros(n_bins)
    bin_indices = (n_bins * y_predicted).astype(np.int)
    # handle corner case when y_predicted = 1.0
    bin_indices[bin_indices == n_bins] = n_bins - 1
    for bin_idx in range(n_bins):
        bin_mask = bin_indices == bin_idx
        mean_probabilities[bin_idx] = y_predicted[bin_mask].sum()
        fraction_of_positives[bin_idx] = y_true[bin_mask].sum()
        bins_size[bin_idx] = bin_mask.sum()
    # To avoid Nan in the normalization
    bins_size[bins_size == 0] = 1
    mean_probabilities /= bins_size
    fraction_of_positives /= bins_size
    calibration_errors = np.abs(fraction_of_positives - mean_probabilities)

    return bins_size, calibration_errors, fraction_of_positives, mean_probabilities


def expected_calibration_error(
    y_true: np.ndarray, y_predicted: np.ndarray, n_bins=10
) -> float:
    """
    Computes the expected calibration error which gives a statistic summary of the
    calibration of a classification model.
    see ref https://www.dbmi.pitt.edu/sites/default/files/Naeini.pdf
    :param y_true:
    :param y_predicted:
    :return:
    """
    bins_size, calibration_errors, _, _ = _calibration_bins(y_true, y_predicted, n_bins)
    return np.sum(bins_size / len(y_true) * calibration_errors)


def maximum_calibration_error(
    y_true: np.ndarray, y_predicted: np.ndarray, n_bins=10
) -> float:
    """
    Computes the maximum calibration error which gives a statistic summary of the
    calibration of a classification model.
    see ref https://www.dbmi.pitt.edu/sites/default/files/Naeini.pdf
    :param y_true:
    :param y_predicted:
    :return:
    """
    bins_size, calibration_errors, _, _ = _calibration_bins(y_true, y_predicted, n_bins)
    return np.max(calibration_errors)


def can_compute_enrichment_factor(dict_metrics: dict) -> bool:
    """Function to check if the metrics available (editable by the user in the Configurator)
    contain precision and proportion_active, both required to add the enrichment factor.
    """
    return ("precision" in dict_metrics) and ("proportion_active" in dict_metrics)


COMMON_METRICS = {
    "mae": {FUNCTION: mean_absolute_error, TYPE: SCORE},
    "precision": {
        FUNCTION: precision_score,
        TYPE: BIN,
        KWARGS: {"zero_division": 0},
    },
    "recall": {FUNCTION: recall_score, TYPE: BIN},
    "proportion_active": {FUNCTION: proportion_active, TYPE: BIN},
}

SCORING_FUNCTIONS = dict(
    COMMON_METRICS,
    **{
        "auc": {FUNCTION: roc_auc_score, TYPE: SCORE},
        "pr_auc": {FUNCTION: pr_auc_score, TYPE: SCORE},
        "kappa": {FUNCTION: cohen_kappa_score, TYPE: BIN},
        "accuracy": {FUNCTION: accuracy_score, TYPE: BIN},
        "f1_score": {
            FUNCTION: f1_score,
            TYPE: BIN,
            KWARGS: {"zero_division": 0},
        },
        "log_loss": {FUNCTION: log_loss, TYPE: SCORE},
        "brier_score_loss": {FUNCTION: brier_score_loss, TYPE: SCORE},
        "expected_calibration_error": {
            FUNCTION: expected_calibration_error,
            TYPE: SCORE,
        },
        "maximum_calibration_error": {FUNCTION: maximum_calibration_error, TYPE: SCORE},
        "uncertainty_auc": {FUNCTION: uncertainty_auc_score, TYPE: BOTH},
    },
)


def score_model(
    y_true_binned,
    prediction_score,
    prediction_threshold=0.5,
    scoring_functions=SCORING_FUNCTIONS,
):
    """
    Returns the metrics computed on the results iterator.
    """
    results = {"prediction_threshold": prediction_threshold}
    prediction = 1 * (np.array(prediction_score) >= prediction_threshold)

    for metric, function in scoring_functions.items():
        if function[TYPE] == SCORE:
            args = (y_true_binned, prediction_score)
        elif function[TYPE] == BIN:
            args = (y_true_binned, prediction)
        elif function[TYPE] == BOTH:
            args = (y_true_binned, prediction, prediction_score)
        else:
            raise ValueError(f"TYPE = {function[TYPE]} not supported")
        try:
            results[metric] = function[FUNCTION](*args, **function.get(KWARGS, {}))
        except Exception as e:
            print(e)
            results[metric] = 1
    if can_compute_enrichment_factor(results):
        results[ENRICHMENT] = results["precision"] / results["proportion_active"]
    return results
