import numpy as np
import pandas as pd

from math import factorial
from itertools import combinations

from .. import Constants as constants
from ..ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel
from ..ModelMapping.TreeNodeComponent import TreeNodeComponent

class NodeSHAPFunctions:
    @staticmethod
    def confidence(class_distribution_with, class_distribution_without):
        def _confidence(class_distribution):
            total = class_distribution.sum()
            if total == 0:
                return 0.0
            return class_distribution.max() / total
        return _confidence(class_distribution_with) - _confidence(class_distribution_without)
    
    @staticmethod
    def gini(class_distribution_with, class_distribution_without):
        def _gini(class_distribution):
            total = class_distribution.sum()
            if total == 0:
                return 0.0
            probabilities = class_distribution / total
            return 1.0 - np.sum(probabilities ** 2)
        return _gini(class_distribution_without) - _gini(class_distribution_with)
    
    @staticmethod
    def entropy(class_distribution_with, class_distribution_without):
        def _entropy(class_distribution):
            total = class_distribution.sum()
            if total == 0:
                return 0.0
            probabilities = class_distribution / total
            probabilities = probabilities[probabilities > 0]
            return -np.sum(probabilities * np.log2(probabilities))
        return _entropy(class_distribution_without) - _entropy(class_distribution_with)
    
    @staticmethod
    def prediction(class_distribution_with, class_distribution_without):
        return float(np.argmax(class_distribution_with) != np.argmax(class_distribution_without))
    
    functions_map = {
        constants.NodeSHAPFunctionType.Confidence: confidence,
        constants.NodeSHAPFunctionType.gini: gini,
        constants.NodeSHAPFunctionType.entropy: entropy,
        constants.NodeSHAPFunctionType.Prediction: prediction,
    }

    def __class_getitem__(_, node_shap_function_type: constants.NodeSHAPFunctionType):
        assert node_shap_function_type in NodeSHAPFunctions.functions_map, f"Unsupported Node SHAP function type: {node_shap_function_type}"
        return NodeSHAPFunctions.functions_map[node_shap_function_type]

def _compute_node_before_distribution(node: TreeNodeComponent, classes: np.ndarray) -> np.ndarray:
    return np.array([node.get_node_class_count().get(current_class, 0) for current_class in classes], dtype=float)

def _compute_after_distributions(mapped_model: ATreeBasedMappedModel, X: pd.DataFrame, y: pd.Series, classes: np.ndarray) -> dict:
    """
    Compute per-node class distributions from the after-data samples.

    Parameters:
        mapped_model: The mapped tree based model.
        X: After-data features.
        y: After-data labels.
        classes: Array of class labels.

    Returns:
        dict: Mapping from node component_index to class distribution (numpy array of shape (num_classes,)).
    """
    node_indicator = mapped_model.get_node_indicator(X)
    y_indices = np.searchsorted(classes, y.values)

    after_distributions = {}
    for node in mapped_model:
        sample_mask = np.asarray(node_indicator[:, node.component_index].todense()).flatten().astype(bool)
        after_distributions[node] = np.bincount(y_indices[sample_mask], minlength=len(classes)).astype(float)

    return after_distributions


def _get_subset_class_distribution(
    node: TreeNodeComponent,
    subset: frozenset,
    right_set: frozenset,
    class_distributions: dict,
) -> np.ndarray:
    """
    Recursively compute the class distribution for a node given an active subset
    and left-right allocation, by walking the tree top-down.

    Parameters:
        node: The current tree node.
        subset: The set of active internal nodes.
        right_set: The set of active internal nodes where the sample goes right.
        class_distributions: Pre-computed cumulative class distributions per node (TreeNodeComponent → array).

    Returns:
        np.ndarray: The class distribution.
    """
    if node.is_terminal():
        return class_distributions[node.component_index]

    if node in subset:
        # Node is active: follow the constrained direction
        child = node.right_child if node in right_set else node.left_child
        return _get_subset_class_distribution(child, subset, right_set, class_distributions)
    else:
        # Node not active: sum both children
        left = _get_subset_class_distribution(node.left_child, subset, right_set, class_distributions)
        right = _get_subset_class_distribution(node.right_child, subset, right_set, class_distributions)
        return left + right


def compute_node_shap_values(
    mapped_model: ATreeBasedMappedModel,
    X: pd.DataFrame,
    y: pd.Series,
    inverse_spectra_map: dict[TreeNodeComponent, int],
    node_shap_function_type: constants.NodeSHAPFunctionType = None,
) -> np.ndarray:
    """
    Compute Node-SHAP values for all samples and all nodes.

    Parameters:
        mapped_model: The mapped tree based model.
        X: The data samples.
        y: The data labels.
        inverse_spectra_map: Mapping from TreeNodeComponent to spectra index,
            used to place results in the correct row of the output matrix.

    Returns:
        np.ndarray: Shape (num_components, num_samples). node_shap_values[spectra_idx, sample_idx].
    """
    classes  = mapped_model.model.classes_
    internal_nodes = set(filter(TreeNodeComponent.is_internal, mapped_model))
    samples_count = len(X)

    after_distributions = _compute_after_distributions(mapped_model, X, y, classes)
    class_distributions = {
        node.component_index: _compute_node_before_distribution(node, classes) + after_distributions[node]
        for node in mapped_model
    }

    current_node_shap_function_type = node_shap_function_type if node_shap_function_type is not None else constants.DEFAULT_NODE_SHAP_FUNCTION_TYPE
    if current_node_shap_function_type == constants.NodeSHAPFunctionType.Criterion:
        current_node_shap_function_type = getattr(constants.NodeSHAPFunctionType, mapped_model.model.criterion)
    current_shap_function = NodeSHAPFunctions[current_node_shap_function_type]

    # Pre-compute sample left-right allocations: for each sample, which internal nodes go right
    sample_right_sets = []
    for sample_index in range(samples_count):
        right_nodes = set()
        for node in internal_nodes:
            if X[node.feature].iloc[sample_index] > node.threshold:
                right_nodes.add(node)
        sample_right_sets.append(frozenset(right_nodes))

    shap_values = np.zeros((len(mapped_model), samples_count))
    root = mapped_model.root
    n = len(internal_nodes)

    for node in internal_nodes:
        other_nodes = internal_nodes - {node}
        for sample_index, sample_right_est in enumerate(sample_right_sets):
            for subset_size in range(n):
                weight = factorial(subset_size) * factorial(n - subset_size - 1) / factorial(n)
                for subset_tuple in combinations(other_nodes, subset_size):
                    subset = frozenset(subset_tuple)

                    right_set_without = sample_right_est & subset
                    right_set_with = sample_right_est & (subset | {node})

                    dist_without = _get_subset_class_distribution(root, subset, right_set_without, class_distributions)
                    dist_with = _get_subset_class_distribution(root, subset | {node}, right_set_with, class_distributions)

                    shap_values[inverse_spectra_map[node], sample_index] += weight * current_shap_function(dist_with, dist_without)

    return shap_values
