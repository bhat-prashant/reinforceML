#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from experiments.expt_util import estimate_performance
from reinforce_ import ReinforceClassifier
from utils_ import save_logbook, save_model

trans_types = ['scaler', 'selector', 'classifier']  # 'unary','extractor',

if __name__ == "__main__":
    datasets = ['heart', ]
    for dataset in datasets:
        for technique in ['ddqn']:
            data = pd.read_csv('../data/{}.csv'.format(dataset))
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            feat = ReinforceClassifier(pop_size=5, generation=1, use_rl=True, rl_technique=technique,
                                       trans_types=trans_types)
            feat.fit(X, y)
            pipeline, logbook = feat.predict()
            save_logbook(logbook)
            save_model(pipeline)
            estimate_performance(feat, feat._hof[0], trans_types='_'.join(feat._trans_types),
                                 expt_type='classification',
                                 rl_technique=feat._rl_technique, dataset_name=dataset)
