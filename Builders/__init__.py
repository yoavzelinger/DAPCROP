from .DecisionTreeClassifierBuilder import build_tree
from sklearn.base import ClassifierMixin
from copy import deepcopy

def get_sklearn_model(X_train, y_train, previous_model=None) -> ClassifierMixin:
    return build_tree(X_train, y_train, model=deepcopy(previous_model))