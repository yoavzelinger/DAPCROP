import os
import sys
import pandas as pd
from itertools import combinations
from collections import defaultdict
from typing import Generator
from collections.abc import Collection

import traceback

from Methodology import *

import Evaluation.TesterConstants as tester_constants
from .DataManagementTools import *
from Evaluation.Builders import *
from Evaluation.metrics import get_accuracy, get_wasted_effort, get_top_k

def get_dataset(directory: str,
                file_name: str,
                file_extension: str = ".csv"
                )-> Dataset:
    source = os.path.join(directory, f"{file_name}{file_extension}")
    return Dataset(source)

def drift_tree(dataset: Dataset,
               candidate_features: Collection[str],
               repair_window_test_sizes: list[float] = tester_constants.REPAIR_WINDOW_TEST_SIZES,
               min_drift_size: int = tester_constants.MIN_DRIFT_SIZE,
               max_drift_size: int = tester_constants.MAX_DRIFT_SIZE
               ):
    """
    Generate a drifted in a multiple features
    """
    match tester_constants.DRIFT_SYNTHESIZING_VERSION:
        case 1:
            drift_generator = dataset.drift_generator
        case 2:
            drift_generator = dataset.drift_generator_v2
        case _:
            raise ValueError(f"Unsupported drift version: {tester_constants.DRIFT_SYNTHESIZING_VERSION} for semi-synthetic drift generation.")

    current_min_drift_size = max(min_drift_size, 1)
    current_max_drift_size = len(candidate_features)
    if max_drift_size > 0:
        current_max_drift_size = min(current_max_drift_size, max_drift_size)
    for repair_window_test_size in repair_window_test_sizes:
        print(f"\tRepair window size: {(repair_window_test_size * tester_constants.REPAIR_PROPORTION * 100):.1f}%")
        dataset.update_repair_window_size(repair_window_test_size)
        for drift_size in range(current_min_drift_size, current_max_drift_size + 1):
            print(f"\t\tDrift size: {drift_size} / {current_max_drift_size} features")
            for drifting_features in combinations(candidate_features, drift_size):
                print(f"\t\t\tDrifting {', '.join(drifting_features)}")
                drifted_features_types = sorted([dataset.feature_types[drifting_feature] for drifting_feature in drifting_features])
                for (X_before, y_before), (X_repair, y_repair), (X_repair_window, y_repair_window), (X_test, y_test), drift_severity_level, drift_description in drift_generator(drifting_features):
                    yield (X_before, y_before), (X_repair, y_repair), (X_repair_window, y_repair_window), (X_test, y_test), (drift_severity_level, drift_description), set(drifting_features), drifted_features_types, drift_size

def get_drifted_nodes(mapped_model: ATreeBasedMappedModel,
                      drifted_features: set[str]
 ) -> dict[str, list[int]]:
    faulty_features_nodes = {true_faulty_feature : [] for true_faulty_feature in drifted_features}
    node: TreeNodeComponent
    for node in mapped_model:
        if node.feature in drifted_features:
            faulty_features_nodes[node.feature].append(node.get_index())
    return faulty_features_nodes

def is_drift_contains_numeric_features(drifted_features_types):
    return any(map(lambda feature_type: feature_type == tester_constants.constants.FeatureType.Numeric, drifted_features_types))

def is_drift_contains_binary_features(drifted_features_types):
    return any(map(lambda feature_type: feature_type == tester_constants.constants.FeatureType.Binary, drifted_features_types))

def get_total_drift_types(drifted_features_types: list[tester_constants.constants.FeatureType]) -> tester_constants.constants.FeatureType:
    drift_contains_numeric_features, drift_contains_binary_features = is_drift_contains_numeric_features(drifted_features_types), is_drift_contains_binary_features(drifted_features_types)
    if drift_contains_numeric_features and drift_contains_binary_features:
        return tester_constants.constants.FeatureType.Mixed.name
    if drift_contains_numeric_features:
        return tester_constants.constants.FeatureType.Numeric.name
    return tester_constants.constants.FeatureType.Binary.name

def get_drifted_features_types_string(drifted_features_types: list[tester_constants.constants.FeatureType]) -> str:
    return ", ".join(map(lambda feature_type: feature_type.name, drifted_features_types))

def group_by_feature(mapped_model: ATreeBasedMappedModel,
                     nodes_indices: list[int]) -> dict[str, list[int]]:
    feature_indices_map: dict[str, list[int]] = defaultdict(list)
    for node in map(mapped_model.__getitem__, nodes_indices):
        feature_indices_map[node.feature if node.is_internal() else "target"].append(node.get_index())
    return feature_indices_map

class EmptyOracleError(Exception):
    """ Raised when the Oracle diagnoser cannot find any faulty nodes. """
    pass

def diagnosis_generator(mapped_model: ATreeBasedMappedModel,
                        X_repair: pd.DataFrame,
                        y_repair: pd.Series,
                        drifted_features: list[str],
                        diagnosers_data: list[dict[str, object]],
                        calculate_oracle_diagnosis: bool = True,
                        calculate_diagnosis_metrics: bool = True,
                        ) -> Generator[tuple[str, list[int], dict[str, object]], None, None]:
    """
    Generate diagnoses from multiple diagnosers.
    
    Parameters:
        mapped_model (ATreeBasedMappedModel): The mapped model.
        X_repair (pd.DataFrame): The repair data features.
        y_repair (pd.Series): The repair data labels.
        diagnosers_data (list[dict[str, object]]): The diagnosers data.
        
    Yields:
        tuple[str, list[int], dict[str, object]]: diagnoser output name, diagnosis, diagnosis_results
    """
    OracleDiagnoserClass = get_diagnoser(Oracle)
    oracle_parameters = {
        "actual_faulty_features": drifted_features
        }
    print(f"\t\t\t\t{Oracle.__name__} Diagnosing")
    oracle_diagnoser: ADiagnoser = OracleDiagnoserClass(mapped_model, X_repair, y_repair, **oracle_parameters)
    oracle_diagnoses = oracle_diagnoser.get_diagnoses()
    if not oracle_diagnoses or not oracle_diagnoses[0]:
        raise EmptyOracleError("Oracle diagnoser found no faulty nodes.")
    true_faulty_nodes_indices: list[int] = oracle_diagnoses[0]

    faulty_features_nodes = group_by_feature(mapped_model, true_faulty_nodes_indices)
    
    if calculate_oracle_diagnosis:
        yield Oracle.__name__, true_faulty_nodes_indices, { 
            tester_constants.DRIFTED_FEATURES_COLUMN_NAME: ", ".join(map(lambda feature: f"{feature}: {faulty_features_nodes[feature]}", faulty_features_nodes))
        }

    drifted_features = list(faulty_features_nodes.keys())

    for diagnoser_data in diagnosers_data:
        diagnoser_class_name = diagnoser_data["class_name"]
        diagnoser_output_name = diagnoser_data.get("output_name", diagnoser_class_name)
        diagnoser_parameters = diagnoser_data.get("parameters")
        diagnoser_class = get_diagnoser(diagnoser_class_name)
        assert diagnoser_parameters is not None, "Diagnoser must have parameters property"

        print(f"\t\t\t\t{diagnoser_output_name} Diagnosing")
        diagnoser: ADiagnoser = diagnoser_class(mapped_model, X_repair, y_repair, **diagnoser_parameters)
        diagnoses: list[list[int]] = diagnoser.get_diagnoses()
        diagnosis = diagnoses[0] if diagnoses else []
        detected_faulty_features = set(group_by_feature(mapped_model, diagnosis).keys())
        
        diagnosis_results = {
            f"{diagnoser_output_name} {tester_constants.FAULTY_FEATURES_NAME_SUFFIX}": ", ".join(map(str, detected_faulty_features)),
            f"{diagnoser_output_name} {tester_constants.DIAGNOSES_NAME_SUFFIX}": ", ".join(map(str, diagnoses)),
            f"{diagnoser_output_name} {tester_constants.WASTED_EFFORT_NAME_SUFFIX}": \
                get_wasted_effort(mapped_model, diagnoses, faulty_features_nodes) if calculate_diagnosis_metrics else 0
        }

        for K in tester_constants.CORRECTLY_IDENTIFIED_K_LIST:
            diagnosis_results[f"{diagnoser_output_name} {tester_constants.CORRECTLY_IDENTIFIED_NAME_SUFFIX_TEMPLATE.format(k=K)}"] = \
                get_top_k(mapped_model, diagnoses, drifted_features, K) * 100 if calculate_diagnosis_metrics else 0
        
        yield diagnoser_output_name, diagnosis, diagnosis_results

def evaluate_retrained_baselines(
    X_before: pd.DataFrame,
    y_before: pd.Series,
    X_repair: pd.DataFrame,
    y_repair: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    post_drift_test_accuracy: float,
) -> dict[str, float] | None:
    X_before_repair = pd.concat([X_before, X_repair]).reset_index(drop=True)
    y_before_repair = pd.concat([y_before, y_repair]).reset_index(drop=True)

    new_all_retrained_model = get_sklearn_model(X_before_repair, y_before_repair)
    new_drift_retrained_model = get_sklearn_model(X_repair, y_repair)
    if new_all_retrained_model is None or new_drift_retrained_model is None:
        return None

    new_all_retrained_accuracy = get_accuracy(new_all_retrained_model, X_test, y_test)
    new_drift_retrained_accuracy = get_accuracy(new_drift_retrained_model, X_test, y_test)

    return {
        f"{tester_constants.NEW_DRIFT_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": new_drift_retrained_accuracy * 100,
        f"{tester_constants.NEW_DRIFT_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": (new_drift_retrained_accuracy - post_drift_test_accuracy) * 100,
        f"{tester_constants.NEW_ALL_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": new_all_retrained_accuracy * 100,
        f"{tester_constants.NEW_ALL_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": (new_all_retrained_accuracy - post_drift_test_accuracy) * 100,
    }

def build_scenario_settings_dict(
    dataset_name: str,
    dataset_size: int,
    mapped_model: ATreeBasedMappedModel,
    repair_window_percentage: float,
    repair_window_size: int,
    drift_size: int,
    drifted_features_types: list[tester_constants.constants.FeatureType],
    drift_severity_level: int,
    drift_description: str,
    pre_drift_accuracy: float,
    post_drift_test_accuracy: float,
    baseline_results: dict[str, float],
) -> dict[str, object]:
    post_drift_test_accuracy_drop = pre_drift_accuracy - post_drift_test_accuracy
    return {
        tester_constants.DATASET_NAME_COLUMN_NAME: dataset_name,
        tester_constants.DATASET_SIZE_COLUMN_NAME: dataset_size,
        tester_constants.TREE_SIZE_COLUMN_NAME: len(mapped_model),
        tester_constants.TREE_FEATURES_COUNT_COLUMN_NAME: len(mapped_model.model_used_features),
        tester_constants.REPAIR_WINDOW_PERCENTAGE_COLUMN_NAME: round(repair_window_percentage, 2),
        tester_constants.REPAIR_WINDOW_SIZE_COLUMN_NAME: repair_window_size,
        tester_constants.DRIFT_SIZE_COLUMN_NAME: drift_size,
        tester_constants.TOTAL_DRIFT_TYPE_COLUMN_NAME: get_total_drift_types(drifted_features_types),
        tester_constants.DRIFT_SEVERITY_LEVEL_COLUMN_NAME: drift_severity_level,
        tester_constants.DRIFTED_FEATURES_TYPES_COLUMN_NAME: get_drifted_features_types_string(drifted_features_types),
        tester_constants.DRIFT_DESCRIPTION_COLUMN_NAME: drift_description,
        tester_constants.ORIGINAL_ACCURACY_COLUMN_NAME: pre_drift_accuracy * 100,
        tester_constants.AFTER_ACCURACY_COLUMN_NAME: post_drift_test_accuracy * 100,
        tester_constants.AFTER_ACCURACY_DECREASE_COLUMN_NAME: post_drift_test_accuracy_drop * 100,
        **baseline_results,
    }

def fixing_generator(
    mapped_model: ATreeBasedMappedModel,
    X_repair: pd.DataFrame,
    y_repair: pd.Series,
    faulty_nodes_indices: list[int],
    fixers_data: list[dict[str, object]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    post_drift_test_accuracy: float,
    diagnoser_output_name: str,
) -> Generator[dict[str, float], None, None]:
    for fixer_data in fixers_data:
        fixer_class_name = fixer_data["class_name"]
        fixer_output_name = fixer_data.get("output_name", fixer_class_name)
        fixer_parameters = fixer_data.get("parameters") or {}
        fixer_class = get_fixer(fixer_class_name)
        fixer: AFixer = fixer_class(
            mapped_model,
            X_repair,
            y_repair,
            faulty_nodes_indices=faulty_nodes_indices,
            X_prior=X_repair,
            y_prior=y_repair,
            **fixer_parameters,
        )
        fixed_model = fixer.fix_model()
        fixed_test_accuracy = get_accuracy(fixed_model, X_test, y_test)
        test_accuracy_bump = fixed_test_accuracy - post_drift_test_accuracy
        diagnoser_fixer_column_prefix = f"{diagnoser_output_name}-{fixer_output_name}"
        yield {
            f"{diagnoser_fixer_column_prefix} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": fixed_test_accuracy * 100,
            f"{diagnoser_fixer_column_prefix} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": test_accuracy_bump * 100,
        }

def diagnosis_and_fixing_generator(
    mapped_model: ATreeBasedMappedModel,
    X_repair: pd.DataFrame,
    y_repair: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    drifted_features: list[str],
    diagnosers_data: list[dict[str, object]],
    fixers_data: list[dict[str, object]],
    post_drift_test_accuracy: float,
    calculate_oracle_diagnosis: bool = True,
    calculate_diagnosis_metrics: bool = True,
) -> Generator[dict[str, object], None, None]:
    for diagnoser_output_name, faulty_nodes_indices, diagnosis_results in diagnosis_generator(
        mapped_model,
        X_repair,
        y_repair,
        drifted_features,
        diagnosers_data,
        calculate_oracle_diagnosis=calculate_oracle_diagnosis,
        calculate_diagnosis_metrics=calculate_diagnosis_metrics,
    ):
        diagnosis_and_fixing_results: dict[str, object] = dict(diagnosis_results)

        for fixer_results in fixing_generator(
            mapped_model=mapped_model,
            X_repair=X_repair,
            y_repair=y_repair,
            faulty_nodes_indices=faulty_nodes_indices,
            fixers_data=fixers_data,
            X_test=X_test,
            y_test=y_test,
            post_drift_test_accuracy=post_drift_test_accuracy,
            diagnoser_output_name=diagnoser_output_name,
        ):
            diagnosis_and_fixing_results.update(fixer_results)

        yield diagnosis_and_fixing_results

def combine_scenario_results(
    scenario_settings: dict[str, object],
    diagnosis_and_fixing_results: dict[str, object],
) -> dict[str, object]:
    return {**scenario_settings, **diagnosis_and_fixing_results}

def format_scenario_exception(exception: Exception, scenario_description: str) -> Exception:
    exception_class = exception.__class__.__name__
    return Exception(
        f"{exception_class} in {scenario_description}:\n "
        f"{exception}:\n "
        f"{''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))}"
        f"Scenario: {scenario_description}"
    )

def run_single_test(directory, file_name, file_extension: str = ".csv", repair_window_test_sizes=tester_constants.REPAIR_WINDOW_TEST_SIZES, min_drift_size=tester_constants.MIN_DRIFT_SIZE, max_drift_size=tester_constants.MAX_DRIFT_SIZE, diagnosers_data: list[dict[str, object]] = tester_constants.DEFAULT_TESTING_DIAGNOSER, fixers_data: list[dict[str, object]] = tester_constants.DEFAULT_TESTING_FIXER):
    dataset = get_dataset(directory, file_name, file_extension=file_extension)

    X_before, y_before = dataset.get_before_concept_data()
    
    sklearn_model = get_sklearn_model(X_before, y_before)
    if sklearn_model is None:
        return
    pre_drift_accuracy = get_accuracy(sklearn_model, *dataset.get_test_data())

    if pre_drift_accuracy < tester_constants.MODEL_MINIMUM_ORIGINAL_ACCURACY:
        # Original model is not good enough
        return
    
    mapped_model = get_mapped_model(sklearn_model, dataset.feature_types, X_before, y_before)
    
    pre_drift_repair_accuracy = get_accuracy(sklearn_model, *dataset.get_repair_data())
    
    for (X_before, y_before), (X_repair_total, y_repair_total), (X_repair, y_repair), (X_test, y_test), (drift_severity_level, drift_description), drifted_features, drifted_features_types, drift_size in drift_tree(dataset, mapped_model.model_used_features, repair_window_test_sizes=repair_window_test_sizes, min_drift_size=min_drift_size, max_drift_size=max_drift_size):
        try:
            if X_repair.empty or X_test.empty:
                continue
            
            post_drift_test_accuracy = get_accuracy(mapped_model.model, X_test, y_test)
            post_drift_test_accuracy_drop = pre_drift_accuracy - post_drift_test_accuracy
            
            if pre_drift_repair_accuracy - get_accuracy(sklearn_model, X_repair, y_repair) < tester_constants.MODEL_MINIMUM_DRIFT_ACCURACY_DROP or post_drift_test_accuracy_drop < tester_constants.MODEL_MINIMUM_DRIFT_ACCURACY_DROP:   # insignificant drift
                continue

            baseline_results = evaluate_retrained_baselines(X_before, y_before, X_repair, y_repair, X_test, y_test, post_drift_test_accuracy)
            if baseline_results is None:
                continue

            scenario_settings = build_scenario_settings_dict(
                dataset_name=file_name,
                dataset_size=len(dataset),
                mapped_model=mapped_model,
                repair_window_percentage=round(dataset.repair_proportion * dataset.repair_window_proportion * 100, 2),
                repair_window_size=dataset.repair_window_size,
                drift_size=drift_size,
                drifted_features_types=drifted_features_types,
                drift_severity_level=drift_severity_level,
                drift_description=drift_description,
                pre_drift_accuracy=pre_drift_accuracy,
                post_drift_test_accuracy=post_drift_test_accuracy,
                baseline_results=baseline_results,
            )

            try:
                for diagnosis_and_fixing_results in diagnosis_and_fixing_generator(
                    mapped_model, X_repair, y_repair, X_test, y_test,
                    list(drifted_features), diagnosers_data, fixers_data,
                    post_drift_test_accuracy=post_drift_test_accuracy,
                ):
                    yield combine_scenario_results(scenario_settings, diagnosis_and_fixing_results)
            except EmptyOracleError:
                continue
        except Exception as e:
            scenario_description = f"{drift_description}, repair window size: {dataset.repair_window_proportion}"
            yield format_scenario_exception(e, scenario_description)

def run_single_test_v2(directory, file_name, file_extension: str = ".csv", repair_window_test_sizes=tester_constants.REPAIR_WINDOW_TEST_SIZES, min_drift_size=tester_constants.MIN_DRIFT_SIZE, max_drift_size=tester_constants.MAX_DRIFT_SIZE, diagnosers_data: list[dict[str, object]] = tester_constants.DEFAULT_TESTING_DIAGNOSER, fixers_data: list[dict[str, object]] = tester_constants.DEFAULT_TESTING_FIXER):
    dataset = get_dataset(directory, file_name, file_extension=file_extension)
    
    for (X_before, y_before), (X_repair_total, y_repair_total), (X_repair, y_repair), (X_test, y_test), (drift_severity_level, drift_description), drifted_features, drifted_features_types, drift_size in drift_tree(dataset, dataset.features, repair_window_test_sizes=repair_window_test_sizes, min_drift_size=min_drift_size, max_drift_size=max_drift_size):
        try:
            sklearn_model = get_sklearn_model(X_before, y_before)
            if sklearn_model is None:
                continue
            pre_drift_accuracy = sklearn_model.best_accuracy

            if pre_drift_accuracy < tester_constants.MODEL_MINIMUM_ORIGINAL_ACCURACY:
                # Original model is not good enough
                continue
            
            mapped_model = get_mapped_model(sklearn_model, dataset.feature_types, X_before, y_before)

            if not all(feature in mapped_model.model_used_features for feature in drifted_features):    # redundant features drifted
                continue

            if X_repair.empty or X_test.empty:
                continue
            
            post_drift_test_accuracy = get_accuracy(mapped_model.model, X_test, y_test)
            post_drift_test_accuracy_drop = pre_drift_accuracy - post_drift_test_accuracy
            
            if pre_drift_accuracy - get_accuracy(mapped_model.model, X_repair, y_repair) < tester_constants.MODEL_MINIMUM_DRIFT_ACCURACY_DROP or post_drift_test_accuracy_drop < tester_constants.MODEL_MINIMUM_DRIFT_ACCURACY_DROP:    # insignificant drift
                continue

            baseline_results = evaluate_retrained_baselines(X_before, y_before, X_repair, y_repair, X_test, y_test, post_drift_test_accuracy)
            if baseline_results is None:
                continue

            scenario_settings = build_scenario_settings_dict(
                dataset_name=file_name,
                dataset_size=len(dataset),
                mapped_model=mapped_model,
                repair_window_percentage=round(dataset.repair_proportion * dataset.repair_window_proportion * 100, 2),
                repair_window_size=dataset.repair_window_size,
                drift_size=drift_size,
                drifted_features_types=drifted_features_types,
                drift_severity_level=drift_severity_level,
                drift_description=drift_description,
                pre_drift_accuracy=pre_drift_accuracy,
                post_drift_test_accuracy=post_drift_test_accuracy,
                baseline_results=baseline_results,
            )

            try:
                for diagnosis_and_fixing_results in diagnosis_and_fixing_generator(
                    mapped_model, X_repair, y_repair, X_test, y_test,
                    list(drifted_features), diagnosers_data, fixers_data,
                    post_drift_test_accuracy=post_drift_test_accuracy,
                ):
                    yield combine_scenario_results(scenario_settings, diagnosis_and_fixing_results)
            except EmptyOracleError:
                continue
        except Exception as e:
            scenario_description = f"{drift_description}, repair window size: {dataset.repair_window_proportion}"
            yield format_scenario_exception(e, scenario_description)

def run_single_test_v3(generator_name: str, repair_window_test_sizes=tester_constants.REPAIR_WINDOW_TEST_SIZES, diagnosers_data: list[dict[str, object]] = tester_constants.DEFAULT_TESTING_DIAGNOSER, fixers_data: list[dict[str, object]] = tester_constants.DEFAULT_TESTING_FIXER):
    """
    Run tests for v3 synthetic MOA drift for a single generator type.
    Iterates over all configs of the given generator, yielding results one by one.
    Diagnosers are skipped; only fixers are evaluated.
    """
    for (X_before, y_before), (X_repair_total, y_repair_total), (X_test, y_test), drift_description, drifted_features, drifted_features_types, drift_size, feature_types in Dataset.drift_generator_v3(generator_name):
        repair_window_test_size = None
        try:
            sklearn_model = get_sklearn_model(X_before, y_before)
            if sklearn_model is None:
                continue
            pre_drift_accuracy = sklearn_model.best_accuracy

            if pre_drift_accuracy < tester_constants.MODEL_MINIMUM_ORIGINAL_ACCURACY:
                continue

            post_drift_test_accuracy = get_accuracy(sklearn_model, X_test, y_test)
            post_drift_test_accuracy_drop = pre_drift_accuracy - post_drift_test_accuracy

            if post_drift_test_accuracy_drop < tester_constants.MODEL_MINIMUM_DRIFT_ACCURACY_DROP:
                continue

            mapped_model = get_mapped_model(sklearn_model, feature_types, X_before, y_before)

            for repair_window_test_size in repair_window_test_sizes:
                window_size = max(1, int(len(X_repair_total) * repair_window_test_size))
                X_repair, y_repair = X_repair_total.iloc[:window_size], y_repair_total.iloc[:window_size]

                if X_repair.empty:
                    continue

                baseline_results = evaluate_retrained_baselines(X_before, y_before, X_repair, y_repair, X_test, y_test, post_drift_test_accuracy)
                if baseline_results is None:
                    continue

                scenario_settings = build_scenario_settings_dict(
                    dataset_name=drift_description,
                    dataset_size=tester_constants.V3_N_SAMPLES,
                    mapped_model=mapped_model,
                    repair_window_percentage=round(repair_window_test_size * tester_constants.REPAIR_PROPORTION * 100, 2),
                    repair_window_size=len(X_repair),
                    drift_size=drift_size,
                    drifted_features_types=drifted_features_types,
                    drift_severity_level=tester_constants.DUMMY_DRIFT_SEVERITY_LEVEL,
                    drift_description=drift_description,
                    pre_drift_accuracy=pre_drift_accuracy,
                    post_drift_test_accuracy=post_drift_test_accuracy,
                    baseline_results=baseline_results,
                )

                try:
                    for diagnosis_and_fixing_results in diagnosis_and_fixing_generator(
                        mapped_model, X_repair, y_repair, X_test, y_test,
                        list(drifted_features), diagnosers_data, fixers_data,
                        post_drift_test_accuracy=post_drift_test_accuracy,
                        calculate_oracle_diagnosis=False,
                        calculate_diagnosis_metrics=False,
                    ):
                        yield combine_scenario_results(scenario_settings, diagnosis_and_fixing_results)
                except EmptyOracleError:
                    continue
        except Exception as e:
            scenario_description = f"{drift_description}, repair window size: {repair_window_test_size}"
            yield format_scenario_exception(e, scenario_description)


def get_example_mapped_model(directory=tester_constants.DATASETS_DIRECTORY_FULL_PATH, file_name=tester_constants.EXAMPLE_FILE_NAME):
    dataset = get_dataset(directory, file_name)
    X_train, y_train = dataset.get_before_concept_data()
    sklearn_model = get_sklearn_model(X_train, y_train)
    return get_mapped_model(sklearn_model, dataset.feature_types, X_train, y_train)

def sanity_run(directory=tester_constants.DATASETS_DIRECTORY_FULL_PATH, file_name=tester_constants.EXAMPLE_FILE_NAME, diagnosers_data=tester_constants.DEFAULT_TESTING_DIAGNOSER):
    for result in run_single_test(directory=directory, file_name=file_name, diagnosers_data=diagnosers_data):
        print(result)
        
if __name__ == "__main__":
    file_name = tester_constants.EXAMPLE_FILE_NAME if len(sys.argv) < 2 else sys.argv[1]
    sanity_run(tester_constants.DATASETS_DIRECTORY_FULL_PATH, file_name)