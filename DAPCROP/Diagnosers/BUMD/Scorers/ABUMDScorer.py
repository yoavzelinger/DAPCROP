from abc import ABC, abstractmethod
from statistics import mean

import pandas as pd

from DAPCROP.ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel
from DAPCROP.ModelMapping.TreeNodeComponent import TreeNodeComponent


class ABUMDScorer(ABC):
    """
    Base class for BUMD per-node scorers.

    Subclasses must implement `get_score(node)`. Scores are computed eagerly in
    the constructor and cached on `self._scores`. Nodes that should be excluded
    from BUMD calculations entirely (e.g. nodes with no after-drift samples)
    must be assigned the sentinel `IRRELEVANT_SCORE`.
    """

    IRRELEVANT_SCORE: float = -1.0

    def __init__(self,
                 mapped_model: ATreeBasedMappedModel,
                 X: pd.DataFrame,
                 y: pd.Series
    ):
        """
        Initialize the scorer and eagerly compute scores for every node.

        Parameters:
            mapped_model (ATreeBasedMappedModel): The mapped model.
            X (DataFrame): The after-drift data.
            y (Series): The after-drift target column.
        """
        self.mapped_model = mapped_model
        self.X_after = X
        self.y_after = y

        self._scores: dict[TreeNodeComponent, float] = {}
        for node in self.mapped_model:
            self._scores[node] = self.get_score(node)

        relevant_leaf_scores = [
            self._scores[leaf]
            for leaf in self.mapped_model.root.get_all_leaves()
            if self.is_relevant(leaf)
        ]
        if relevant_leaf_scores:
            self._leaves_mean = mean(relevant_leaf_scores)
        else:
            self._leaves_mean = 0.0

    @abstractmethod
    def get_score(self,
                  node: TreeNodeComponent
     ) -> float:
        """
        Compute the fault score of a node.

        Return `ABUMDScorer.IRRELEVANT_SCORE` if the node should be excluded
        from all BUMD calculations.

        Parameters:
            node (TreeNodeComponent): The node to score.

        Returns:
            float: The fault score, or `IRRELEVANT_SCORE`.
        """
        ...

    def is_relevant(self,
                    node: TreeNodeComponent
     ) -> bool:
        """
        Whether the node has a usable score.

        Parameters:
            node (TreeNodeComponent): The node.

        Returns:
            bool: True iff the node's score is not the irrelevance sentinel.
        """
        return self._scores[node] != ABUMDScorer.IRRELEVANT_SCORE

    def is_faulty(self,
                  node: TreeNodeComponent
     ) -> bool:
        """
        Whether the node is considered faulty.

        Default rule: relevant nodes whose score is at least the mean of
        relevant-leaf scores. Using `>=` rather than `>` guarantees that the
        faulty set is non-empty whenever at least one relevant leaf exists
        (the max-scoring leaf always satisfies `>= mean`).

        Parameters:
            node (TreeNodeComponent): The node.

        Returns:
            bool: True iff the node is considered faulty.
        """
        if not self.is_relevant(node):
            return False
        return self._scores[node] >= self._leaves_mean

    def get_faulty_leaves(self) -> set[TreeNodeComponent]:
        """
        Get the set of leaves considered faulty.

        Returns:
            set[TreeNodeComponent]: The faulty leaves.
        """
        return {leaf for leaf in self.mapped_model.root.get_all_leaves() if self.is_faulty(leaf)}

    def should_merge(self,
                     candidate_substitutor: TreeNodeComponent,
                     *candidates_substitutee: TreeNodeComponent
     ) -> bool:
        """
        Whether to replace the substitutees by the substitutor in the diagnosis.

        Default rule: substitutor's score is strictly greater than the mean
        of the substitutees' scores. Returns False if any involved node is
        irrelevant.

        Parameters:
            candidate_substitutor (TreeNodeComponent): The merging candidate (typically the parent).
            candidates_substitutee (TreeNodeComponent): The candidates to be replaced (typically the faulty siblings).

        Returns:
            bool: True iff the merge should be performed.
        """
        if not self.is_relevant(candidate_substitutor):
            return False
        if any(not self.is_relevant(c) for c in candidates_substitutee):
            return False
        substitutor_score = self._scores[candidate_substitutor]
        substitutees_mean = mean(self._scores[c] for c in candidates_substitutee)
        return substitutor_score > substitutees_mean
