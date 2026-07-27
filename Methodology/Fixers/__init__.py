from ..APPETITE.Fixers import *
from ..APPETITE import Fixers as appetite_fixers

from .. import Constants as constants

from .ForestFixerWrapper import ForestFixerWrapper
from .DependentForestFixerWrapper import DependentForestFixerWrapper

def get_fixer(fixer_name: str):
    FixerClass = appetite_fixers.get_fixer(fixer_name)
    match constants.DRIFTING_MODEL_TYPE:
        case constants.DriftingModelType.Tree:
            return FixerClass
        case constants.DriftingModelType.Forest:
            match constants.DRIFTING_MODEL:
                case constants.DriftingModel.RandomForest:
                    FixerWrapperClass = ForestFixerWrapper
                case constants.DriftingModel.AdaBoost:
                    FixerWrapperClass = DependentForestFixerWrapper
            return lambda *args, **kwargs: FixerWrapperClass(
                FixerClass,
                *args, **kwargs
            )

__all__ = appetite_fixers.__all__