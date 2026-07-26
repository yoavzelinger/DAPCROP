from .ATreeBasedMappedModel import ATreeBasedMappedModel
from .TreeNodeComponent import TreeNodeComponent
from .MappedDecisionTree import MappedDecisionTree

def get_mapped_model(sklearn_model, feature_types, X, y):
    return MappedDecisionTree(sklearn_model, feature_types=feature_types, X=X, y=y)

__all__ = [
    "ATreeBasedMappedModel",
    "TreeNodeComponent",
    "MappedDecisionTree",
    "get_mapped_model"
]