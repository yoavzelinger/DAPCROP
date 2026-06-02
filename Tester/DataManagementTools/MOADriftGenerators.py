from itertools import permutations, islice

from river.datasets.synth import SEA, Agrawal, ConceptDriftStream

import Tester.TesterConstants as tester_constants


def _build_sea_scenarios():
    """All ordered pairs of SEA variants (12 configs)."""
    scenario_functions = []
    for before_concept_variant, after_concept_variant in permutations(tester_constants.MOA_SEA_VARIANTS, 2):
        scenario_functions.append((
            f"SEA(variant={before_concept_variant}) -> SEA(variant={after_concept_variant})",
            lambda position, before_variant=before_concept_variant, after_variant=after_concept_variant: zip(*islice(
                ConceptDriftStream(stream=SEA(variant=before_variant), drift_stream=SEA(variant=after_variant), position=position, width=20),
                tester_constants.V3_N_SAMPLES))
        ))
    return scenario_functions


def _build_agrawal_scenarios():
    """All ordered pairs of Agrawal functions (90 configs)."""
    scenario_functions = []
    for agrawal_before_function, agrawal_after_function in permutations(tester_constants.MOA_AGRAWAL_FUNCTIONS, 2):
        scenario_functions.append((
            f"Agrawal(function={agrawal_before_function}) -> Agrawal(function={agrawal_after_function})",
            lambda position, before_variant=agrawal_before_function, after_variant=agrawal_after_function: zip(*islice(
                ConceptDriftStream(stream=Agrawal(classification_function=before_variant), drift_stream=Agrawal(classification_function=after_variant), position=position, width=20),
                tester_constants.V3_N_SAMPLES))
        ))
    return scenario_functions


# Dictionary: generator_name -> tuple of (list of (description, create_stream) tuples, feature_types)
MOA_GENERATOR_CONFIGS = {
    tester_constants.MOA_SEA_GENERATOR_NAME: (_build_sea_scenarios(), tester_constants.MOA_SEA_FEATURE_TYPES, ),
    tester_constants.MOA_AGRAWAL_GENERATOR_NAME: (_build_agrawal_scenarios(), tester_constants.MOA_AGRAWAL_FEATURE_TYPES, ),
}
