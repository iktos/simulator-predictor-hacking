MODELS_DIR = "MODELS_DO_NOT_MODIFY_90percent"
MODEL_NAME = "model_for_optim"
RESCORER_NAME = "model_rescorer"
RESCORER_DATA_NAME = "model_rescorer_data"
RESCORER_DATA_ENSEMBLE_NAME = "model_rescorer_data_ensemble"
MODEL_CONTROL = "model_control"
DATA_CONTROL = "data_control"
SIMILARITY = "similarity"
QED_NAME = "QED"
NOISE_LEVELS = [0, 20, 40, 60, 80]
COMPUTED_TARGETS = [QED_NAME, SIMILARITY]
DEFAULT_RESCORING = [SIMILARITY]
SCORING_WO_MODEL = [SIMILARITY]
SEP = "_SEP_"
QED_THRESHOLD = 0.7
ORACLE_MAPPING = {f"{QED_NAME}_{noise}": QED_NAME for noise in NOISE_LEVELS}
COMPUTED_ORACLES = COMPUTED_TARGETS + list(ORACLE_MAPPING.keys())


opt_args = {}
opt_args["graph_ga"] = dict(
    smi_file="./data/guacamol_v1_train.smiles",
    population_size=100,
    offspring_size=200,
    generations=100,
    mutation_rate=0.01,
    n_jobs=-1,
    random_start=True,
    patience=150,
    canonicalize=False,
)

opt_args["lstm_hc"] = dict(
    pretrained_model_path="./guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_0.473.pt",
    n_epochs=151,
    mols_to_sample=1028,
    keep_top=512,
    optimize_n_epochs=1,
    max_len=100,
    optimize_batch_size=64,
    number_final_samples=1028,
    sample_final_model_only=False,
    random_start=True,
    smi_file="./data/guacamol_v1_train.smiles",
    n_jobs=-1,
    canonicalize=False,
)
