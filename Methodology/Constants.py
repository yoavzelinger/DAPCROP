from .APPETITE.Constants import *

from enum import Enum, auto

class DriftingModel(Enum):
    DecisionTree = auto()
    RandomForest = auto()
    AdaBoost = auto()

DRIFTING_MODEL = DriftingModel.RandomForest

class DriftingModelType(Enum):
    Tree = auto()
    Forest = auto()

DRIFTING_MODEL_TYPE: DriftingModelType
match DRIFTING_MODEL:
    case DriftingModel.DecisionTree:
        DRIFTING_MODEL_TYPE = DriftingModelType.Tree
    case DriftingModel.RandomForest | DriftingModel.AdaBoost:
        DRIFTING_MODEL_TYPE = DriftingModelType.Forest

DISTRIBUTE_DIAGNOSES_COMPUTATION = True
DISTRIBUTE_FIXING_COMPUTATION = False
