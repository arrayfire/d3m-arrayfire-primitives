"""
af_primitives

"""
__version__ = '0.1.0'
__author__ = 'ArrayFire'
__all__ = [
    "af_KNClassifier",
    "af_LogisticRegression",
    "afSKImputer",
    "afSKStringImputer",
    "afSKMLPClassifier",
    "afSKMLPRegressor",
]

## sub-packages
from . import af_KNClassifier
from . import af_LogisticRegression
from . import afSKImputer
from . import afSKStringImputer
from . import afSKMLPClassifier
from . import afSKMLPRegressor

