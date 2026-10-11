import os
import importlib
import inspect

from .ABUMDScorer import ABUMDScorer

SCORER_CLASSES_DICT: dict[str, type[ABUMDScorer]] = {}


def _load_scorer_classes():
    """
    Load all classes in this package that inherit from ABUMDScorer.
    """
    base_dir = os.path.dirname(__file__)
    base_pkg = __name__

    for filename in os.listdir(base_dir):
        if not filename.endswith(".py") or filename == "__init__.py":
            continue
        module_name = filename[:-3]
        full_module_name = f"{base_pkg}.{module_name}"

        module = importlib.import_module(full_module_name)
        for _, cls in inspect.getmembers(
            module,
            lambda c: inspect.isclass(c) and issubclass(c, ABUMDScorer) and c is not ABUMDScorer
        ):
            SCORER_CLASSES_DICT[cls.__name__] = cls


_load_scorer_classes()


def get_scorer_class(scorer_name: str) -> type[ABUMDScorer]:
    """
    Get a scorer class by its name.

    Parameters:
        scorer_name (str): The name of the scorer class.

    Returns:
        type[ABUMDScorer]: The scorer class.

    Raises:
        AssertionError: If the scorer class is not found.
    """
    assert scorer_name in SCORER_CLASSES_DICT, f"Scorer {scorer_name} is not supported"
    return SCORER_CLASSES_DICT[scorer_name]


__all__ = ["ABUMDScorer", "SCORER_CLASSES_DICT", "get_scorer_class"]
