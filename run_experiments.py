import random
from pathlib import Path
import time
import copy

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, jaccard_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.ensemble import RakelD
from sklearn.exceptions import ConvergenceWarning

import warnings
import sys
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# todo: comment and describe code better


# load raw data
def load_data(file_path):
    X = []
    y = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        sentence = ' '.join(line.split()[0:-1])
        X.append(sentence)
        labels = line.split()[-1].split('_')
        y.append(labels)
    X = np.array(X, dtype='object')
    y = np.array(y, dtype='object')
    return X, y


# extract resulst from all runs
def calulate_results(scores):
    grouped = scores.groupby('Emotion')
    mean_scores = grouped.agg(
        ovr_f1_mean=('ovr_f1', 'mean'),
        ovr_accuracy_mean=('ovr_accuracy', 'mean'),
        rakel_f1_mean=('rakel_f1', 'mean'),
        rakel_accuracy_mean=('rakel_accuracy', 'mean'),
    )
    mean_scores = mean_scores.append(mean_scores.mean().rename('Overall'))
    var_scores = grouped.agg(
        ovr_f1_var=('ovr_f1', 'var'),
        ovr_accuracy_var=('ovr_accuracy', 'var'),
        rakel_f1_var=('rakel_f1', 'var'),
        rakel_accuracy_var=('rakel_accuracy', 'var'),
    )
    var_scores = var_scores.append(var_scores.mean().rename('Overall'))
    return mean_scores, var_scores


# check segnificance
def calculate_significance(scores, labels):
    results = []
    for label in labels:
        result = {}

        # calculate p-values for F1
        ovr_f1 = scores[scores['Emotion'] == label]['ovr_f1'].values
        rakel_f1 = scores[scores['Emotion'] == label]['rakel_f1'].values
        welch_f1 = ttest_ind(ovr_f1, rakel_f1, equal_var=False, alternative='greater')
        result['f1'] = welch_f1.pvalue

        # calculate p-calues for Accuracy
        ovr_accuracy = scores[scores['Emotion'] == label]['ovr_accuracy'].values
        rakel_accuracy = scores[scores['Emotion'] == label]['rakel_accuracy'].values
        welch_accuracy = ttest_ind(ovr_accuracy, rakel_accuracy, equal_var=False, alternative='greater')
        result['accuracy'] = welch_accuracy.pvalue

        results.append(result)
    return pd.DataFrame(results, index=labels)


# custom scorer
def score_by_label(y, y_pred, label):
    scores = f1_score(y, y_pred, average=None)
    return scores[label]


# initial seed
SEED = 23249425
random.seed(SEED)
# set 10 seeds for randomization of models
local_seeds = {run: random.randint(1, 2**32 - 1) for run in range(10)}

# load data
DATA_PATH = Path(__file__).parent / 'data'
X_train, train_labels = load_data(DATA_PATH / 'train.txt')
X_test, test_labels = load_data(DATA_PATH / 'test.txt')

# binarize target labels
multi_label_binarizer = MultiLabelBinarizer()
y_train = multi_label_binarizer.fit_transform(train_labels)
y_test = multi_label_binarizer.transform(test_labels)
labels = multi_label_binarizer.classes_

# One-vs-Rest Classifyer
ovr_pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(stop_words='english')),
    ('tf', TfidfTransformer()),
    ('svc', LinearSVC(dual=False,
                      max_iter=10e4,
                      random_state=SEED)
     ),
])

ovr_best_params = {}
# find best hyperparams for OvR by label (extimator)
for idx in range(len(labels)):
    ovr_grid = {
        'estimator__tf__use_idf': [True, False],
        'estimator__tf__sublinear_tf': [True, False],
        'estimator__svc__C': [0.1, 1, 10, 100, 1000],
        'estimator__svc__penalty': ['l1', 'l2'],
        'estimator__svc__class_weight': [None, 'balanced'],
    }
    ovr = OneVsRestClassifier(ovr_pipeline)
    ovr_grid_search = GridSearchCV(ovr,
                                   param_grid=ovr_grid,
                                   scoring=make_scorer(score_by_label, label=idx),
                                   refit=True,
                                   n_jobs=-1)
    ovr_grid_search.fit(X_train, y_train)
    ovr_best_params[labels[idx]] = ovr_grid_search.best_params_

# Benchmark
results_all_runs = []
times_all_runs = []
for run, seed in local_seeds.items():
    print('Creating OvR Classifier with best parameters')
    ovr = OneVsRestClassifier(ovr_pipeline)
    ovr.estimators_ = [copy.deepcopy(ovr_pipeline) for i in range(len(labels))]
    for idx in range(len(labels)):
        params = ovr_best_params[labels[idx]]
        ovr.estimators_[idx].set_params(svc__C=params['estimator__svc__C'],
                                        svc__penalty=params['estimator__svc__penalty'],
                                        svc__class_weight=params['estimator__svc__class_weight'],
                                        tf__sublinear_tf=params['estimator__tf__sublinear_tf'],
                                        tf__use_idf=params['estimator__tf__use_idf'],
                                        svc__random_state=seed)

    ovr_start_time = time.time()
    ovr.fit(X_train, y_train)
    ovr_prediction = ovr.predict(X_test)
    ovr_stop_time = time.time()

    ovr_f1 = f1_score(y_test, ovr_prediction, average=None)
    ovr_accuracy = jaccard_score(y_test, ovr_prediction, average=None)

    # RAKEL Classifyer
    rakel_pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(stop_words='english')),
        ('tf-idf_log', TfidfTransformer(sublinear_tf=True)),
        ('rakel', RakelD(
            base_classifier=LinearSVC(C=1, random_state=seed, class_weight='balanced'),
            base_classifier_require_dense=[True, True],
            labelset_size=3
            )
         )
    ])

    rakel_start_time = time.time()
    rakel_pipeline.fit(X_train, y_train)
    rakel_prediction = rakel_pipeline.predict(X_test)
    rakel_stop_time = time.time()
    rakel_f1 = f1_score(y_test, rakel_prediction, average=None)
    rakel_accuracy = jaccard_score(y_test, rakel_prediction, average=None)

    # capture results in DataFrame
    result_run = pd.DataFrame({
        'Emotion': labels,
        'run': run,
        'ovr_f1': ovr_f1,
        'ovr_accuracy': ovr_accuracy,
        'rakel_f1': rakel_f1,
        'rakel_accuracy': rakel_accuracy,
    })
    results_all_runs.append(result_run)

    # capture times
    times_all_runs.append((run + 1, ovr_stop_time - ovr_start_time, rakel_stop_time - rakel_start_time))

# get and extract relevant outputs
results = pd.concat(results_all_runs)
mean_scores, var_scores = calulate_results(results)
p_values = calculate_significance(results, labels)
times = pd.DataFrame.from_records(times_all_runs,
                                  columns=['run', 'ovr_sec', 'rakel_sec']
                                  ).round(3)
seeds = pd.DataFrame.from_dict(local_seeds,
                               orient='index',
                               columns=['seed'])

# save results
OUTPUT_PATH = Path(__file__).parent / 'output'
mean_scores.to_csv(OUTPUT_PATH / 'mean_scores.csv')
mean_scores.to_latex(OUTPUT_PATH / 'mean_scores.txt', float_format='%.3f')
var_scores.to_csv(OUTPUT_PATH / 'var_scores.csv')
var_scores.to_latex(OUTPUT_PATH / 'var_scores.txt', float_format='%.3f')
p_values.to_csv(OUTPUT_PATH / 'p_values.csv')
p_values.to_latex(OUTPUT_PATH / 'p_values.txt', float_format='%.3f')
times.to_csv(OUTPUT_PATH / 'times.csv', index=False)
times.to_latex(OUTPUT_PATH / 'times.txt', index=False, float_format='%.3f')
seeds.to_csv(OUTPUT_PATH / 'seeds.csv', index_label='run')
seeds.to_latex(OUTPUT_PATH / 'seeds.txt', index=False)
