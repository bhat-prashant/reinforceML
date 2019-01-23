# -*- coding: utf-8 -*-

import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from tpot import TPOTClassifier
from deap import creator
from base_utils import BaseFeatureEngineer

class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, *args, **kwargs):
        super(FeatureEngineer, self).__init__( *args, **kwargs)











