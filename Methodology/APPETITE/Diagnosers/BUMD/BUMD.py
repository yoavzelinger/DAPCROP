from collections import defaultdict
from statistics import mean

import pandas as pd

from ...ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel
from ...ModelMapping.TreeNodeComponent import TreeNodeComponent

from ..ADiagnoser import ADiagnoser

from .Scorers import get_scorer_class, ABUMDScorer


# BUMD - Bottom-Up Merge Diagnoser
class BUMD(ADiagnoser):
    """
    Bottom-Up Merge Diagnoser.

    Algorithm:
      1. Compute a pluggable per-node fault score via an `ABUMDScorer`.
      2. Seed the diagnosis with all faulty leaves.
      3. Sweep the depth buckets bottom-up: for every pair of faulty siblings at
         the current depth, ask the scorer whether to replace them with their
         parent. If yes, swap them out for the parent in the diagnosis (and the
         parent enters the bucket of its own depth, where it can merge again).
      4. Terminate when no populated bucket has depth <= current_depth.

    Yields a single candidate diagnosis (a length-1 ranked list).
    """

    def __init__(self,
                 mapped_model: ATreeBasedMappedModel,
                 X: pd.DataFrame,
                 y: pd.Series,
                 scorer_name: str,
                 scorer_parameters: dict = {}
    ):
        """
        Initialize the BUMD diagnoser.

        Parameters:
            mapped_model (ATreeBasedMappedModel): The mapped model.
            X (DataFrame): The after-drift data.
            y (Series): The after-drift target column.
            scorer_name (str): The name of the `ABUMDScorer` subclass to use.
            scorer_parameters (dict): Constructor parameters for the scorer.
        """
        super().__init__(mapped_model, X, y)
        ScorerClass = get_scorer_class(scorer_name)
        self.scorer: ABUMDScorer = ScorerClass(mapped_model, X, y, **scorer_parameters)

    def _build_diagnosis(self) -> tuple[list[int], float]:
        faulty_set: set[TreeNodeComponent] = self.scorer.get_faulty_leaves()
        
        if not faulty_set:
            return [], 0.0
        
        for current_depth in range(max(map(lambda node: node.depth, faulty_set)), -1, -1):
            if current_depth < min(map(lambda node: node.depth, faulty_set)):
                break

            current_depth_faulty_nodes = list(filter(lambda node: node.depth == current_depth, faulty_set))

            if not any(current_depth_faulty_nodes):
                continue

            parent_siblings_map: dict[TreeNodeComponent, list[TreeNodeComponent]] = defaultdict(list)
            for node in current_depth_faulty_nodes:
                parent_siblings_map[node.parent].append(node)

            for parent, siblings in parent_siblings_map.items():
                if parent is None or len(siblings) < 2:
                    continue

                left, right = tuple(siblings)

                if self.scorer.should_merge(parent, left, right):
                    faulty_set.discard(left)
                    faulty_set.discard(right)
                    faulty_set.add(parent)

        diagnosis_indices = [node.get_index() for node in faulty_set]
        diagnosis_score = mean([self.scorer._scores[node] for node in faulty_set])
        return diagnosis_indices, diagnosis_score

    def get_diagnoses(self,
                      retrieve_ranks: bool = False
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        """
        Get the single BUMD diagnosis (wrapped in a length-1 ranked list).

        Parameters:
            retrieve_ranks (bool): Whether to return the diagnosis rank.

        Returns:
            list[list[int]] | list[tuple[list[int], float]]: The diagnosis.
        """
        if self.diagnoses is None:
            diagnosis_indices, diagnosis_score = self._build_diagnosis()
            self.diagnoses = [(diagnosis_indices, diagnosis_score)]
        return super().get_diagnoses(retrieve_ranks)
