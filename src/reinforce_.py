#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from base_ import BaseFeatureEngineer

class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, *args, **kwargs):
        super(FeatureEngineer, self).__init__( *args, **kwargs)











