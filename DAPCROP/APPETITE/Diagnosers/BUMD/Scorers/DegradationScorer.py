from copy import deepcopy

from ....ModelMapping.TreeNodeComponent import TreeNodeComponent

from .ABUMDScorer import ABUMDScorer


class DegradationScorer(ABUMDScorer):
    """
    Scorer that measures per-node accuracy degradation between before- and
    after-drift data.

    Score = max(0, before_accuracy - after_accuracy).

    Empty-bucket policy:
      - If the node has no samples reaching it before the drift, its
        before-accuracy is 0 (the value `confidence` already holds in that case).
      - If the node has no samples reaching it after the drift, return
        `IRRELEVANT_SCORE` (the node cannot have participated in any
        misclassification, hence cannot be considered faulty).
    """

    def _get_before_accuracy(self,
                             node: TreeNodeComponent
     ) -> float:
        return node.confidence

    def _get_after_accuracy(self,
                            node: TreeNodeComponent
     ) -> float | None:
        node_after = deepcopy(node)
        node_after.update_node_data_attributes(self.X_after, self.y_after)
        if node_after.reached_samples_count == 0:
            return None
        return node_after.confidence

    def get_score(self,
                  node: TreeNodeComponent
     ) -> float:
        after_accuracy = self._get_after_accuracy(node)
        if after_accuracy is None:
            return ABUMDScorer.IRRELEVANT_SCORE
        before_accuracy = self._get_before_accuracy(node)
        return max(0.0, before_accuracy - after_accuracy)
