import random
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, jaccard_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.ensemble import RakelD

# todo: VisibleDeprecationWarning
# skmultilearn/cluster/random.py:129: VisibleDeprecationWarning:
# Creating an ndarray from ragged nested sequences
# (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or
#  shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray

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
    var_scores = grouped.agg(
        ovr_f1_var=('ovr_f1', 'var'),
        ovr_accuracy_var=('ovr_accuracy', 'var'),
        rakel_f1_var=('rakel_f1', 'var'),
        rakel_accuracy_var=('rakel_accuracy', 'var'),
    )
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


# initial seed
SEED = 23249425
random.seed(SEED)

# load data
DATA_PATH = Path(__file__).parent / 'data'
X_train, train_labels = load_data(DATA_PATH / 'train.txt')
X_test, test_labels = load_data(DATA_PATH / 'test.txt')

# binarize target labels
multi_label_binarizer = MultiLabelBinarizer()
y_train = multi_label_binarizer.fit_transform(train_labels)
y_test = multi_label_binarizer.transform(test_labels)
labels = multi_label_binarizer.classes_

# run Experiments
seeds_all_runs = []
results_all_runs = []
times_all_runs = []

for n in range(10):
    # randomize Models
    local_seed = random.randint(1, 2**32 - 1)
    seeds_all_runs.append((n + 1, local_seed))

    # One-vs-Rest Classifyer
    # todo: Grid search per label
    ovr_pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer()),  # todo: check if vocabulary needed or not
        ('tf-idf', TfidfTransformer()),
        # todo: optional oversampling
        ('ovr', OneVsRestClassifier(
            LinearSVC(penalty='l1',
                      dual=False,
                      max_iter=10e3,
                      random_state=local_seed)
            )
         ),
    ])

    ovr_start_time = time.time()
    ovr_pipeline.fit(X_train, y_train)
    ovr_prediction = ovr_pipeline.predict(X_test)
    ovr_stop_time = time.time()
    ovr_f1 = f1_score(y_test, ovr_prediction, average=None)
    ovr_accuracy = jaccard_score(y_test, ovr_prediction, average=None)

    # RAKEL Classifyer
    rakel_pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(stop_words='english')),  # todo: check if better without stop_words and if vocabulary needed
        ('tf-idf_log', TfidfTransformer(sublinear_tf=True)),
        # todo: oversampling
        ('rakel', RakelD(
            base_classifier=LinearSVC(C=1, random_state=local_seed),
            base_classifier_require_dense=[True, True],  # todo: change to False to test spares data
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
        'run': n,
        'ovr_f1': ovr_f1,
        'ovr_accuracy': ovr_accuracy,
        'rakel_f1': rakel_f1,
        'rakel_accuracy': rakel_accuracy,
    })
    results_all_runs.append(result_run)

    # capture times
    times_all_runs.append((n + 1, ovr_stop_time - ovr_start_time, rakel_stop_time - rakel_start_time))

# get and extract relevant results
results = pd.concat(results_all_runs)
mean_scores, var_scores = calulate_results(results)
p_values = calculate_significance(results, labels)
times = pd.DataFrame.from_records(times_all_runs,
                                  columns=['run', 'ovr_sec', 'rakel_sec']
                                  ).round(3)
seeds = pd.DataFrame.from_records(seeds_all_runs,
                                  columns=['run', 'seed'])

# print results
# print(mean_results)
# print(times)
# print(p_values)

# save results
RESULTS_PATH = Path(__file__).parent / 'results'
mean_scores.to_csv(RESULTS_PATH / 'mean_scores.csv')
mean_scores.to_latex(RESULTS_PATH / 'mean_scores.txt', float_format='%.3f')
var_scores.to_csv(RESULTS_PATH / 'var_scores.csv')
var_scores.to_latex(RESULTS_PATH / 'var_scores.txt', float_format='%.3f')
p_values.to_csv(RESULTS_PATH / 'p_values.csv')
p_values.to_latex(RESULTS_PATH / 'p_values.txt', float_format='%.3f')
times.to_csv(RESULTS_PATH / 'times.csv', index=False)
times.to_latex(RESULTS_PATH / 'times.txt', index=False, float_format='%.3f')
seeds.to_csv(RESULTS_PATH / 'seeds.csv', index=False)
seeds.to_latex(RESULTS_PATH / 'seeds.txt', index=False)
