import pandas as pd

from DAPCROP.ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel
from DAPCROP.ModelMapping.TreeNodeComponent import TreeNodeComponent

from .ABUMDScorer import ABUMDScorer


class DiagnoserScorer(ABUMDScorer):
    """
    Scorer that wraps another `ADiagnoser` and uses its per-node ranks as the
    fault scores.

    The wrapped diagnoser must yield singular diagnoses (each diagnosis is a
    list containing exactly one node index). Nodes the wrapped diagnoser does
    not score (e.g. STAT skips terminal leaves) get `IRRELEVANT_SCORE`.
    """

    def __init__(self,
                 mapped_model: ATreeBasedMappedModel,
                 X: pd.DataFrame,
                 y: pd.Series,
                 diagnoser_name: str,
                 diagnoser_parameters: dict = {}
    ):
        """
        Initialize the scorer by running the inner diagnoser once.

        Parameters:
            mapped_model (ATreeBasedMappedModel): The mapped model.
            X (DataFrame): The after-drift data.
            y (Series): The after-drift target column.
            diagnoser_name (str): The name of the inner diagnoser class.
            diagnoser_parameters (dict): Constructor parameters for the inner diagnoser.
        """
        # Lazy import to avoid a circular import at package-load time:
        # `Diagnosers/__init__.py` populates `DIAGNOSER_CLASSES_DICT` by walking
        # the package, including this module.
        from DAPCROP.Diagnosers import get_diagnoser

        DiagnoserClass = get_diagnoser(diagnoser_name)
        scorer_diagnoser = DiagnoserClass(mapped_model, X, y, **diagnoser_parameters)
        ranked_diagnoses = scorer_diagnoser.get_diagnoses(retrieve_ranks=True)

        assert all(len(diagnosis) == 1 for diagnosis, _ in ranked_diagnoses), (
            f"DiagnoserScorer requires a diagnoser that yields singular diagnoses, "
            f"but {diagnoser_name} returned at least one multi-node diagnosis."
        )

        self._nodes_scores: dict[int, float] = {
            diagnosis[0]: rank for diagnosis, rank in ranked_diagnoses
        }

        super().__init__(mapped_model, X, y)

    def get_score(self,
                  node: TreeNodeComponent
     ) -> float:
        return self._nodes_scores.get(node.get_index(), ABUMDScorer.IRRELEVANT_SCORE)
