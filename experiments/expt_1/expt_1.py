#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import csv
import os

import pandas as pd

from reinforce_ import ReinforceFeatureEngineer


def estimate_performance(reinforce, individual, trans_types, rl_technique, dataset_name='temp'):
    """ Temporary: For ongoing research paper

    :param individual:
    :return:
    """
    reinforce._estimator.fit(reinforce._X_train, reinforce._y_train)
    y_pred = reinforce._estimator.predict(reinforce._X_val)
    initial_score = reinforce._scorer(reinforce._y_val, y_pred)
    final_score, _ = reinforce._evaluate(individual)
    score = ((1 - initial_score) - (1 - final_score)) / (1 - initial_score)
    filename = '../../results/expt_1/result_{}.csv'.format(reinforce._estimator.__class__.__name__)
    rows = []
    if os.path.exists(filename):
        write_type = 'a'
    else:
        write_type = 'w'
        rows.append(['dataset', 'Use_RL', 'RL_technique', 'transformer_types', 'initial_score', 'final_score',
                     'improvement (%)'])
    with open(filename, write_type) as csvFile:
        writer = csv.writer(csvFile)
        rows.append([dataset_name, "RL={}".format(str(reinforce._use_rl)), rl_technique, trans_types,
                     "{0:.3f}".format(initial_score),
                         "{0:.3f}".format(final_score), "{0:.3f}".format(score)])
        writer.writerows(rows)
    csvFile.close()


if __name__ == "__main__":
    datasets = ['heart', ]  # 'wind', 'puma_8', 'puma_32'
    for dataset in datasets:
        data = pd.read_csv('../../data/{}.csv'.format(dataset))
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        transformer_types = ['unary', 'scaler', 'extractor']
        feat = ReinforceFeatureEngineer(pop_size=40, generation=10, use_rl=True, trans_types=transformer_types,
                                        rl_technique='dqn')
        feat.fit(X, y)
        pipeline = feat.predict()
        estimate_performance(feat, feat._hof[0], trans_types='_'.join(transformer_types),
                             rl_technique=feat._rl_technique, dataset_name=dataset)
