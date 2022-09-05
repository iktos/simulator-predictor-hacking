from guacamol_baselines.graph_ga.goal_directed_generation import GB_GA_Generator
from guacamol_baselines.smiles_lstm_hc.smiles_rnn_directed_generator import (
    SmilesRnnDirectedGenerator,
)


OPTIMIZER_FACTORY = {
    "graph_ga": GB_GA_Generator,
    "lstm_hc": SmilesRnnDirectedGenerator,
}
