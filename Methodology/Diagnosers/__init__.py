from ..APPETITE.Diagnosers import *
from ..APPETITE import Diagnosers as appetite_diagnosers

from .. import Constants as constants

from .DistributedForestDiagnoser import DistributedForestDiagnoser

def get_diagnoser(diagnoser: str | type[appetite_diagnosers.Oracle]):
    DiagnoserClass = appetite_diagnosers.get_diagnoser(diagnoser)
    match constants.DRIFTING_MODEL_TYPE:
        case constants.DriftingModelType.Tree:
            return DiagnoserClass
        case constants.DriftingModelType.Forest:
            if not constants.DISTRIBUTE_DIAGNOSES_COMPUTATION:
                return DiagnoserClass
            return lambda *args, **kwargs: DistributedForestDiagnoser(
                DiagnoserClass,
                *args, **kwargs
            )

__all__ = appetite_diagnosers.__all__ + ["get_diagnoser"]