import pandas as pd
from copy import deepcopy

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from DOSPINER.ModelMapping.MappedDecisionTree import MappedDecisionTree

from .ForestFixerWrapper import ForestFixerWrapper


class DependentForestFixerWrapper(ForestFixerWrapper):
    """
    A fixer wrapper for AdaBoost classifiers that fixes trees iteratively.
    
    Unlike ForestFixer fixing (which can be parallelized), in here the fixing is sequential
    because each fix depends on the previous one. When fixing a tree, samples that reached
    faulty nodes in the last fixed tree are excluded from the current fix.
    """
    alias = "dependent_forest_fixer_wrapper"

    def _get_samples_reaching_faulty_nodes(self,
                                           mapped_estimator: MappedDecisionTree,
                                           faulty_node_indices: list[int],
                                           X: pd.DataFrame
    ) -> pd.Index:
        """
        Get the indices of samples that reach any faulty node in the given estimator.
        
        Args:
            mapped_estimator: The mapped decision tree estimator
            faulty_node_indices: List of faulty node indices in this estimator
            X: The data to check
            
        Returns:
            pd.Index: Indices of samples that reached any faulty node
        """
        samples_reaching_fault = set()
        for node_index in faulty_node_indices:
            faulty_node = mapped_estimator[node_index]
            reached_X = faulty_node.get_data_reached_node(X, allow_empty=True)
            samples_reaching_fault.update(reached_X.index)
        return pd.Index(list(samples_reaching_fault))

    def fix_model(self) -> AdaBoostClassifier | RandomForestClassifier:
        """
        Fix the model using iterative processing.
        
        Each tree is fixed sequentially. When fixing a tree, samples that reached
        faulty nodes in the last fixed tree are excluded from the current fix.

        Returns:
            AdaBoostClassifier | RandomForestClassifier: The fixed ensemble model.
        """
        self.fixed_model: AdaBoostClassifier | RandomForestClassifier = deepcopy(self.sklearn_model)
        
        faulty_estimator_indices = sorted(self.faulty_estimators.keys())
        
        last_fixed_mapped_estimator: MappedDecisionTree | None = None
        
        for estimator_index in faulty_estimator_indices:
            mapped_estimator = self.original_mapped_model.mapped_estimators[estimator_index]
            
            # Exclude samples that reached faulty nodes in the last fixed tree
            mask = pd.Series(True, index=self.X.index)
            if last_fixed_mapped_estimator is not None:
                mask = ~self.X.index.isin(
                    self._get_samples_reaching_faulty_nodes(
                        last_fixed_mapped_estimator,
                        self.faulty_estimators[estimator_index],
                        self.X
                    )
                )
                
            fixer = self.base_fixer_class(
                mapped_estimator,
                self.X[mask],
                self.y[mask],
                self.faulty_estimators[estimator_index],
                self.X_prior,
                self.y_prior,
                **self.base_fixer_parameters
            )
            self.fixed_model.estimators_[estimator_index] = fixer.fix_model()
                        
            last_fixed_mapped_estimator = mapped_estimator

        return self.fixed_model
