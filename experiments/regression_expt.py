#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import warnings

from sklearn.datasets import load_boston

warnings.filterwarnings("ignore")
from experiments.expt_util import estimate_performance, save_logbook
from reinforce_ import ReinforceRegressor

if __name__ == "__main__":
    datasets = ['heart', ]
    for dataset in datasets:
        for technique in ['dqn']:
            X, y = load_boston(return_X_y=True)
            feat = ReinforceRegressor(pop_size=20, generation=5, use_rl=False, rl_technique=technique)
            feat.fit(X, y)
            pipeline, logbook = feat.predict()
            save_logbook(logbook)
            estimate_performance(feat, feat._hof[0], trans_types='_'.join(feat._trans_types), expt_type='regression',
                                 rl_technique=feat._rl_technique, dataset_name=dataset)
