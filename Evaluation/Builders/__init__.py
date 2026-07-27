from Evaluation.SingleModel.Builders import get_sklearn_model as build_tree
from .ForestClassifierBuilder import build_forest
from copy import deepcopy
from Methodology import constants

def get_sklearn_model(X_train, y_train, previous_model=None):
    match constants.DRIFTING_MODEL_TYPE:
        case constants.DriftingModelType.Tree:
            return build_tree(X_train, y_train, previous_model)
        case constants.DriftingModelType.Forest:
            return build_forest(X_train, y_train, model=deepcopy(previous_model))
        case _:
            raise ValueError(f"Unsupported drifting model: {constants.DRIFTING_MODEL_TYPE}")

__all__ = ["get_sklearn_model"]