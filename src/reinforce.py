# -*- coding: utf-8 -*-
from multiprocessing import cpu_count
from sklearn.externals.joblib import Parallel, delayed, Memory
from base_utils import BaseFeatureEngineer

class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, *args, **kwargs):
        super(FeatureEngineer, self).__init__( *args, **kwargs)











