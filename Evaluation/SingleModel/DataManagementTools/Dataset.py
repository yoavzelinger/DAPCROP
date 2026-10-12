import os
from scipy.io.arff import loadarff

import pandas as pd
import math

from typing import Generator, Callable
from collections.abc import Collection

from .. import TesterConstants as tester_constants

from .DriftSimulation import single_feature_concept_drift_generator, multiple_features_concept_drift_generator
from .MOADriftGenerators import MOA_GENERATOR_CONFIGS


class Dataset:
    partitions = ["before", "after", "repair", "test"]

    def __init__(self, 
                 source: str | pd.DataFrame
    ):
        """
        source (str | DataFrame): If str, the path to the dataset file; If DataFrame, the dataset itself.
        """
        # Get data
        if type(source) == str:    # Path to the file
            self.name, source_format = os.path.splitext(os.path.basename(source))
            if source_format in (".csv", ".data", ".txt"):
                source = pd.read_csv(source)
            elif source_format == ".arff":
                data, _ = loadarff(source)
                source = pd.DataFrame(data)
        assert isinstance(source, pd.DataFrame)

        self.target_name = source.columns[-1]
        self.data, y = self.split_features_targets(source)

        self.feature_types = {}
        one_hot_encoded_dict = {}
        for column_name in self.data.columns:
            column_type = self.data[column_name].dtype
            if column_type not in [object, bool]:   # Numeric
                self.data[column_name] = self.data[column_name].fillna(self.data[column_name].mean())    # Fill NaN values
                self.feature_types[column_name] = tester_constants.constants.FeatureType.Numeric
                continue
            # Categorical or Binary
            self.data[column_name] = self.data[column_name].fillna(self.data[column_name].mode().iloc[0])    # Fill NaN values
            if len(self.data[column_name].unique()) <= 2: # Consider as binary
                column_type = bool
            if column_type == bool or not tester_constants.ONE_HOT_ENCODING_CATEGORICAL:
                self.data[column_name] = pd.Categorical(self.data[column_name])
                self.data[column_name] = self.data[column_name].cat.codes
                self.feature_types[column_name] = tester_constants.constants.FeatureType.Binary if column_type == bool else tester_constants.constants.FeatureType.Categorical
                continue
            # One hot encoding with multiple values
            one_hot_encoded_dict.update({f"{column_name}_{value}": tester_constants.constants.FeatureType.Binary for value in self.data[column_name].unique()})
            self.data = pd.get_dummies(self.data, columns=[column_name])
        self.feature_types.update(one_hot_encoded_dict)
        self.features: list[str] = list(self.feature_types.keys())

        self.data[self.target_name] = pd.Categorical(y.fillna(y.mode().iloc[0]))
        self.data[self.target_name] = self.data[self.target_name].cat.codes

        if tester_constants.DRIFT_SYNTHESIS_TYPE in (tester_constants.DriftSynthesisType.FEATURE_SHIFT, 1):  # shuffle data - same shuffle always
            self.data = self.data.sample(frac=1, random_state=tester_constants.constants.RANDOM_STATE).reset_index(drop=True)

        self.data.attrs["name"] = self.name

        n_samples = len(self.data)
        self.before_proportion, self.repair_proportion, self.test_proportion = tester_constants.PROPORTIONS_TUPLE
        self.before_concept_size = math.floor(self.before_proportion*n_samples)
        self.total_repair_size = math.floor(self.repair_proportion*n_samples)
        self.test_size = math.floor(self.test_proportion*n_samples)
        self.after_concept_size = self.total_repair_size + self.test_size

        assert all([0 < current_size for current_size in (self.before_concept_size, self.total_repair_size, self.test_size)])
        assert (self.before_concept_size + self.total_repair_size + self.test_size) <= n_samples

        self.update_repair_window_size(tester_constants.DEFAULT_REPAIR_WINDOW_PROPORTION)

    def __len__(self) -> int:
        return len(self.data)

    def update_repair_window_size(self, 
                          new_repair_window_size: int | float
     ) -> None:
        assert 0 < new_repair_window_size <= 1

        self.repair_window_proportion = new_repair_window_size
        self.repair_window_size = math.floor(self.total_repair_size * self.repair_window_proportion)

    def split_features_targets(self, 
                               data: pd.DataFrame
     ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split the data to X and y
        
        Parameters:
            data (DataFrame): The data to split
        
        Returns:
            tuple[DataFrame, Series]: The X and y
        """
        X = data.drop(columns=[self.target_name]).reset_index(drop=True)
        y = data[self.target_name].reset_index(drop=True)
        return X, y

    def get_before_concept_data(self) -> tuple[pd.DataFrame, pd.Series]:
        before_concept_data = self.data.iloc[:self.before_concept_size]
        return self.split_features_targets(before_concept_data)
    
    def get_after_concept_data(self) -> tuple[pd.DataFrame, pd.Series]:
        after_concept_data = self.data.iloc[self.before_concept_size:]
        return self.split_features_targets(after_concept_data)
    
    def get_repair_data(self) -> tuple[pd.DataFrame, pd.Series]:
        X_after, y_after = self.get_after_concept_data()
        return X_after.iloc[:self.total_repair_size], y_after.iloc[:self.total_repair_size]
    
    def get_test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        X_after, y_after = self.get_after_concept_data()
        return X_after.iloc[-self.test_size:], y_after.iloc[-self.test_size:]

    def _drift_data_generator(self,
                   data: pd.DataFrame,
                   drift_features: str | list[str],
                   severity_levels: tuple = tester_constants.DEFAULT_GENERATED_SEVERITY_LEVELS
     ) -> Generator[pd.DataFrame, None, None]:
        """
        Create a drift in the data
        
        Parameters:
            data (DataFrame): The data to drift
            drift_features (str or list): single feature or list of features to drift
            severity_levels (tuple[int]): The severity levels the column should be drifted to. Default is all.
                
        Returns:
            DataFrame: The drifted data
        """
        if type(drift_features) == str:
            assert drift_features in data.columns, f"Feature {drift_features} not in the dataset"
            feature_type = self.feature_types[drift_features]
            return single_feature_concept_drift_generator(data, drift_features, feature_type, severity_levels)
        assert all([feature in data for feature in drift_features]), "Not all features in the dataset"
        # Get subset of the dictionary
        drift_features_dict = {feature: self.feature_types[feature] for feature in drift_features}
        return multiple_features_concept_drift_generator(data, drift_features_dict, severity_levels)

    def feature_shift_drift_generator(self,
                        drift_features: str | Collection[str],
                        severity_levels: tuple[int, ...] = tester_constants.DEFAULT_GENERATED_SEVERITY_LEVELS
     ) -> Generator[tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], int, str], None, None]:
        """
        Feature-shift drift generator for the after concept partition.
        
        Parameters:
            drift_features (str or Collection[str]): Single feature or collection of features to drift.
            severity_levels (tuple[int, ...]): The severity levels the column should be drifted to. Default is all.

        Returns:
            Generator yielding:
                (X_before, y_before), (X_repair, y_repair), (X_repair_window, y_repair_window), (X_test, y_test), drift_severity_level, drift_description
        """
        X_before, y_before = self.get_before_concept_data()
        original_X, y = self.get_after_concept_data()
        for drifted_X, drift_severity_level, drift_description in self._drift_data_generator(original_X, drift_features, severity_levels):
            drift_description += f"_window[{self.repair_proportion * self.repair_window_proportion * 100:.1f}%]"
            # split to repair and test
            X_repair, y_repair = drifted_X.iloc[:self.total_repair_size], y.iloc[:self.total_repair_size]
            X_repair_window, y_repair_window = drifted_X.iloc[:self.repair_window_size], y.iloc[:self.repair_window_size]
            X_test, y_test = drifted_X.iloc[self.test_size:], y.iloc[self.test_size:]
            yield (X_before, y_before), (X_repair, y_repair), (X_repair_window, y_repair_window), (X_test, y_test), drift_severity_level, drift_description

    # Backward compatibility alias
    drift_generator = feature_shift_drift_generator

    def get_sorted_data_by_features(self,
                              features: set[str]
     ) -> None:
        """
        Sort the data by the given features.
        
        Parameters:
            features (list[str]): The features to sort by.
        """
        if len(features) == 2:
            raise NotImplementedError("Sorting by more than one feature is not implemented yet")
        
        feature = next(iter(features))
        
        assert self.feature_types[feature] == tester_constants.constants.FeatureType.Numeric, "Sorting by non-numeric feature is not supported"
        
        # sort by the feature
        return self.data.sort_values(by=feature, ascending=True).reset_index(drop=True)

    def distribution_edge_drift_generator(self,
                           drift_features: str | Collection[str],
                           severity_levels: tuple[int, ...] = tester_constants.DUMMY_GENERATED_SEVERITY_LEVELS
        ) -> Generator[tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], int, str], None, None]:
        """
        Distribution-edge drift generator for a specific partition.
        
        The drift is done by sorting the data by the given features and splitting the data to before and after concept.
        The post-drift will be on the top or bottom of the sorted data.
        The before concept and after concept will be shuffled before re-partitioning to repair and test.

        Parameters:
            drift_features (str or Collection[str]): The features to drift.
            severity_levels (tuple[int, ...]): Dummy placeholder severity levels.
        Returns:
            Generator yielding:
                (X_before, y_before), (X_repair, y_repair), (X_repair_window, y_repair_window), (X_test, y_test), drift_severity_level, drift_description
        """
        if isinstance(drift_features, str):
            drift_features = [drift_features]
        if not any(self.feature_types[feature] == tester_constants.constants.FeatureType.Numeric for feature in drift_features):
            return

        sorted_data = self.get_sorted_data_by_features(set(drift_features))

        drift_type_mapping: dict[str, Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]] = {
            "top": lambda data: (data.iloc[: self.before_concept_size], data.iloc[-self.after_concept_size: ]), # after is the highest values
            "bottom": lambda data: (data.iloc[-self.before_concept_size: ], data.iloc[: self.after_concept_size]) # after is the lowest values
        }

        for drift_type, drift_function in drift_type_mapping.items():
            drift_description = f"{drift_type} {'-'.join(sorted(drift_features))}"
            before_concept_data, after_concept_data = drift_function(sorted_data)
            assert len(before_concept_data) == self.before_concept_size and len(after_concept_data) == self.after_concept_size, "Drifted data size is not correct"
            
            before_concept_data = before_concept_data.sample(frac=1, random_state=tester_constants.constants.RANDOM_STATE).reset_index(drop=True)
            after_concept_data = after_concept_data.sample(frac=1, random_state=tester_constants.constants.RANDOM_STATE).reset_index(drop=True)

            X_before, y_before = self.split_features_targets(before_concept_data)
            X_repair, y_repair = self.split_features_targets(after_concept_data.iloc[ :self.total_repair_size])
            X_repair_window, y_repair_window = self.split_features_targets(after_concept_data.iloc[ :self.repair_window_size])
            X_test, y_test = self.split_features_targets(after_concept_data.iloc[-self.test_size: ])

            for drift_severity_level in severity_levels:
                yield (X_before, y_before), (X_repair, y_repair), (X_repair_window, y_repair_window), (X_test, y_test), drift_severity_level, drift_description

    # Backward compatibility alias
    drift_generator_v2 = distribution_edge_drift_generator

    @staticmethod
    def moa_stream_drift_generator(generator_name: str
    ) -> Generator[tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], str], None, None]:
        """
        Generate synthetic concept drift data using MOA stream generators for a given generator type.

        Yields for each scenario:
            (X_before, y_before), (X_repair_full, y_repair_full), (X_test, y_test),
            description, drifted_features, drifted_features_types, drift_size, feature_types
        """

        position = int(tester_constants.V3_N_SAMPLES * tester_constants.BEFORE_PROPORTION)
        repair_size = int(tester_constants.V3_N_SAMPLES * tester_constants.REPAIR_PROPORTION)
        test_size = int(tester_constants.V3_N_SAMPLES * tester_constants.TEST_PROPORTION)

        scenario_functions, feature_types = MOA_GENERATOR_CONFIGS[generator_name]

        for description, scenario_function in scenario_functions:
            X, y = scenario_function(position)
            X, y = pd.DataFrame(X), pd.Series(y, name="target")

            assert set(feature_types.keys()) == set(X.columns), f"Unexpected synthetic features. Expected: {set(feature_types.keys())}, actual: {set(X.columns)}"

            drifted_features = set(X.columns.tolist())
            drifted_features_types = sorted(feature_types.values())
            drift_size = len(drifted_features)

            X_before, y_before = X.iloc[:position], y.iloc[:position]
            X_repair_full, y_repair_full = X.iloc[position:position + repair_size], y.iloc[position:position + repair_size]
            X_test, y_test = X.iloc[-test_size:], y.iloc[-test_size:]

            yield (X_before, y_before), (X_repair_full, y_repair_full), (X_test, y_test), \
                    description, drifted_features, drifted_features_types, drift_size, feature_types

    # Backward compatibility alias
    drift_generator_v3 = moa_stream_drift_generator