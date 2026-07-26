from Evaluation.SingleModel.TesterConstants import *
from Evaluation.SingleModel.TesterConstants import _CRITERIONS
from Evaluation.SingleModel.TesterConstants import CORRECTLY_IDENTIFIED_K_LIST as TREE_CORRECTLY_IDENTIFIED_K_LIST
from Methodology import constants

FOREST_CORRECTLY_IDENTIFIED_K_LIST = [1, 3, 5]

match constants.DRIFTING_MODEL_TYPE:
    case constants.DriftingModelType.Tree:
        CORRECTLY_IDENTIFIED_K_LIST = TREE_CORRECTLY_IDENTIFIED_K_LIST
    case constants.DriftingModelType.Forest:
        CORRECTLY_IDENTIFIED_K_LIST = FOREST_CORRECTLY_IDENTIFIED_K_LIST

_FOREST_NUM_ESTIMATORS = 30
FOREST_PARAM_GRID = {
    "criterion": _CRITERIONS,
    "n_estimators": [_FOREST_NUM_ESTIMATORS]
}