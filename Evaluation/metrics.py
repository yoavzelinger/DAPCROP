from collections import defaultdict
from Methodology import *
from Evaluation.SingleModel.metrics import *
from Evaluation.SingleModel.metrics import get_current_correctly_identified_ratio as base_get_current_correctly_identified_ratio

def get_current_correctly_identified_ratio(mapped_model: ATreeBasedMappedModel,
                                           diagnosis: list[int],
                                           faulty_features: set[str]
                                           ) -> float:
    if isinstance(mapped_model, MappedForest):
        group_by_tree_map = defaultdict(list)
        for node_index in diagnosis:
            estimator_index, _ = mapped_model.component_estimator_map[node_index]
            group_by_tree_map[estimator_index].append(node_index)
        diagnosis_divided_to_trees = list(group_by_tree_map.values())
        
        correctly_identified_ratio_sums: float = 0
        for tree_diagnosis in diagnosis_divided_to_trees:
            correctly_identified_ratio_sums += base_get_current_correctly_identified_ratio(mapped_model, tree_diagnosis, faulty_features)
        
        return correctly_identified_ratio_sums / len(diagnosis_divided_to_trees)
        
    return base_get_current_correctly_identified_ratio(mapped_model, diagnosis, faulty_features)