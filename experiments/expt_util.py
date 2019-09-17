#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import csv
import datetime
import os

now = datetime.datetime.now()


def estimate_performance(reinforce, individual, trans_types, rl_technique, dataset_name='temp', expt_type=''):
    """ Temporary: For ongoing research paper

    :param individual:
    :return:
    """
    reinforce._estimator.fit(reinforce._X_train, reinforce._y_train)
    y_pred = reinforce._estimator.predict(reinforce._X_val)
    initial_score = reinforce._scorer(reinforce._y_val, y_pred)
    final_score, _ = reinforce._evaluate(individual)
    score = ((1 - initial_score) - (1 - final_score)) / (1 - initial_score)
    filename = '../results/result_{}_{}.csv'.format(expt_type, reinforce._estimator.__class__.__name__)
    rows = []
    if os.path.exists(filename):
        write_type = 'a'
    else:
        write_type = 'w'
        rows.append(['time', 'dataset', 'Use_RL', 'RL_technique', 'transformer_types', 'initial_score', 'final_score',
                     'improvement (%)'])
    with open(filename, write_type) as csvFile:
        writer = csv.writer(csvFile)
        rows.append([now.strftime("%Y-%m-%d_%H:%M"), dataset_name, "RL={}".format(str(reinforce._use_rl)), rl_technique,
                     trans_types,
                     "{0:.3f}".format(initial_score),
                     "{0:.3f}".format(final_score), "{0:.3f}".format(score)])
        writer.writerows(rows)
    csvFile.close()

