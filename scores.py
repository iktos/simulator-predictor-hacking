from typing import Dict, List

from predictions import get_model_providers
from utils import TPScoringFunction, MultiObjScoringFunction
from constants import (
    MODEL_NAME,
    MODEL_CONTROL,
    RESCORER_NAME,
)


def scoring_factory(
    chid: str, model_type: str, target_names: List[str]
) -> Dict[str, MultiObjScoringFunction]:
    predictive_models, computed_scores = get_model_providers(
        chid, target_names, model_type
    )

    predictive_models_scoring_functions = {
        MODEL_NAME: [
            TPScoringFunction(
                model_provider.target_name, model_provider.get_model_optim()
            )
            for model_provider in predictive_models
        ],
        MODEL_CONTROL: [
            TPScoringFunction(
                model_provider.target_name, model_provider.get_model_control()
            )
            for model_provider in predictive_models
        ],
        # DATA_CONTROL: [
        #     TPScoringFunction(model_provider.target_name, model_provider.get_data_control()) for model_provider in predictive_models
        # ],
        RESCORER_NAME: [
            TPScoringFunction(model_provider.target_name, model_provider.get_rescorer())
            for model_provider in predictive_models
        ],
        # RESCORER_DATA_NAME: [
        #     TPScoringFunction(model_provider.target_name,
        #                       model_provider.get_rescorer_data()) for model_provider in
        #     predictive_models
        # ],
        # RESCORER_DATA_ENSEMBLE_NAME: [
        #     TPScoringFunction(model_provider.target_name,
        #                       model_provider.get_rescorer_data_ensemble()) for model_provider in
        #     predictive_models
        # ],
    }
    return {
        key: MultiObjScoringFunction(
            predictive_scores
            + [computed_score.get_model() for computed_score in computed_scores]
        )
        for key, predictive_scores in predictive_models_scoring_functions.items()
    }
