from ..APPETITE.ModelMapping import *
from ..APPETITE import ModelMapping as appetite_get_decision_tree_mapping

from .. import Constants as constants

from .MappedForest import MappedForest

def get_mapped_model(sklearn_model, feature_types, X, y):
    match constants.DRIFTING_MODEL_TYPE:
        case constants.DriftingModelType.Tree:
            return appetite_get_decision_tree_mapping.get_mapped_model(sklearn_model, feature_types, X, y)
        case constants.DriftingModelType.Forest:
            return MappedForest(sklearn_model, feature_types=feature_types, X=X, y=y)
        case _:
            raise ValueError(f"Unsupported drifting model: {constants.DRIFTING_MODEL_TYPE}")

__all__ = appetite_get_decision_tree_mapping.__all__ + ["MappedForest"]