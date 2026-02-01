from copy import deepcopy
from sklearn.metrics import accuracy_score

from DOSPINER import *

import Tester.TesterConstants as tester_constants

def get_accuracy(model, X, y):
    y_predicted = model.predict(X)
    return accuracy_score(y, y_predicted)

def get_correctly_identified_ratio(detected_faulty_features: set[str],
                                    true_faulty_features: set[str]
                                    ) -> float:
    identified_features = map(lambda feature: feature in detected_faulty_features, true_faulty_features)
    return sum(identified_features) / len(true_faulty_features)

def get_wasted_effort(mapped_model: ATreeBasedMappedModel,
                      diagnoses: list[list[int]],
                      faulty_features_nodes: dict[str, list[int]]
 ) -> float:
    """
    Calculate the wasted effort of the diagnoser.
    In here the wasted effort is calculated for fixes of all the nodes that include a faulty feature.

    Parameters:
    mapped_model (ATreeBasedMappedModel): The mapped model.
    diagnoses (list[list[int]]): The diagnoses of the nodes.
    faulty_features_nodes (dict[str, list[int]]): Dict of the drifted features and their corresponding faulty nodes indices.
    require_full_fix (bool): If True, the diagnoser is required to fix all faulty nodes of a feature

    Returns:
    float: The wasted effort (can be float in case we fix randomly with float expected value).
    """
    undetected_faulty_features_nodes = deepcopy(faulty_features_nodes)
    wasted_effort_nodes = set()
    are_all_faults_detected = lambda: not any(undetected_faulty_features_nodes.values())
    for diagnosis in diagnoses:
        for diagnosed_faulty_node in map(mapped_model.__getitem__, diagnosis):
            diagnosed_faulty_feature = diagnosed_faulty_node.feature
            if diagnosed_faulty_feature not in undetected_faulty_features_nodes: # a wasted effort
                wasted_effort_nodes.add(diagnosed_faulty_node)
            else:   # relevant fix
                if tester_constants.WASTED_EFFORT_REQUIRE_FULL_FIX and diagnosed_faulty_node.get_index() in undetected_faulty_features_nodes[diagnosed_faulty_feature]:
                    undetected_faulty_features_nodes[diagnosed_faulty_feature].remove(diagnosed_faulty_node.get_index())
                else:
                    undetected_faulty_features_nodes[diagnosed_faulty_feature] = []
        if are_all_faults_detected():
            return len(wasted_effort_nodes)
        
    # Didn't detect all the faulty features, handle all the missing nodes

    current_wasted_effort = len(wasted_effort_nodes)
    current_undetected_faults_count = sum(map(len, undetected_faulty_features_nodes.values()))

    healthy_nodes_count = len(mapped_model) - sum(map(len, faulty_features_nodes.values()))
    assert healthy_nodes_count >= len(wasted_effort_nodes), "Wasted effort nodes count is greater than the healthy nodes count (suppose to be subset of)"
    current_undetected_wasted_effort = healthy_nodes_count - current_wasted_effort
        
    handling_missing_action = {
        "all": healthy_nodes_count,
        "none": current_wasted_effort,  # (+ 0)
        "random": current_wasted_effort + (current_undetected_faults_count * current_undetected_wasted_effort) / (current_undetected_faults_count + 1)
    }
            
    wasted_effort = handling_missing_action[tester_constants.WASTED_EFFORT_MISSING_ACTION]
    
    if tester_constants.NORMALIZE_WASTED_EFFORT:
        # normalize by total number of components
        wasted_effort = wasted_effort / len(mapped_model) * 100
    
    return wasted_effort