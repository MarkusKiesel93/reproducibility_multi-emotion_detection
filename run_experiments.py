import numpy as np
import pandas as pd
import random
from pathlib import Path
import time
import copy
from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, jaccard_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.ensemble import RakelD
import warnings
import sys
import os


# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


# find best hyperparams for One-vs-Rest model by underlying estimator
# one estimator is built by label (class)
def ovr_hyperparameter_optimization(X, y, labels, seed):
    best_params = {}

    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer()),
        ('tf', TfidfTransformer()),
        ('svc', LinearSVC(dual=False,
                          max_iter=10e4,
                          random_state=seed)
         ),
    ])

    for idx in range(len(labels)):
        grid = {
            'tf__use_idf': [True, False],
            'tf__sublinear_tf': [True, False],
            'svc__C': [0.1, 1, 10, 100, 1000],
            'svc__penalty': ['l1', 'l2'],
            'svc__class_weight': [None, 'balanced'],
        }
        grid_search = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring=make_scorer(f1_score),
            refit=True,
            cv=StratifiedKFold(n_splits=5),
            n_jobs=-1)

        # extract current label at index
        y_current_label = [label[idx] for label in y]
        grid_search.fit(X, y_current_label)
        best_params[labels[idx]] = grid_search.best_params_

    return best_params


# One-vs-Rest model: training, predicition and extracting of scores
def ovr_model(X_train, X_test, y_train, y_test, labels, seed, params=None):
    # set parameters for pipeline for each estimator of the OvR model
    if params is not None:
        pipeline = Pipeline([
            ('count_vectorizer', CountVectorizer()),
            ('tf', TfidfTransformer()),
            ('svc', LinearSVC(
                dual=False,
                max_iter=10e4,
                random_state=seed)
             ),
        ])
        ovr = OneVsRestClassifier(pipeline)
        ovr.estimators_ = [copy.deepcopy(pipeline) for i in range(len(labels))]
        for idx in range(len(labels)):
            ovr.estimators_[idx].set_params(**params[labels[idx]])
    # use simpler model where each estimator of the OvR model is equal
    else:
        pipeline = Pipeline([
            ('count_vectorizer', CountVectorizer()),
            ('tf', TfidfTransformer(sublinear_tf=True)),
            ('svc', LinearSVC(
                C=0.1,
                penalty='l1',
                class_weight='balanced',
                dual=False,
                max_iter=10e4,
                random_state=seed)
             ),
        ])
        ovr = OneVsRestClassifier(pipeline)
    # train and predict model
    start_time = time.time()
    ovr.fit(X_train, y_train)
    prediction = ovr.predict(X_test)
    stop_time = time.time()
    # calculate scores
    f1 = f1_score(y_test, prediction, average=None)
    accuracy = jaccard_score(y_test, prediction, average=None)

    return f1, accuracy, stop_time - start_time


# RAKEL model: training, predicition and extracting of scores
def rakel_model(X_train, X_test, y_train, y_test, labels, seed):
    rakel = Pipeline([
        ('count_vectorizer', CountVectorizer()),
        ('tf-idf_log', TfidfTransformer(sublinear_tf=True)),
        ('rakel', RakelD(
            base_classifier=LinearSVC(C=1,
                                      class_weight='balanced',
                                      random_state=seed),
            base_classifier_require_dense=[True, True],
            labelset_size=3
            )
         )
    ])
    # train and predict model
    start_time = time.time()
    rakel.fit(X_train, y_train)
    prediction = rakel.predict(X_test)
    stop_time = time.time()
    # calculate scores
    f1 = f1_score(y_test, prediction, average=None)
    accuracy = jaccard_score(y_test, prediction, average=None)

    return f1, accuracy, stop_time - start_time


# load raw data from text files
def load_data(file_path):
    X = []
    y = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        # sentence is everything except the last word
        sentence = ' '.join(line.split()[0:-1])
        # labels are the last word on each line in form "label1_label2_..."
        labels = line.split()[-1]
        # add empty lists for None labels
        if labels != 'None':
            labels = labels.split('_')
        else:
            labels = []
        X.append(sentence)
        y.append(labels)
    X = np.array(X, dtype='object')
    y = np.array(y, dtype='object')
    return X, y


# calculate mean scores and variances over all runs
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


# calculate significance levels using Welch's one-sided t-test
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


if __name__ == '__main__':
    # if True: simple model has given equal parameters for all OvR estimators
    # if False: grid search used to find best parameters per estimator
    simple_model = False

    # load data
    DATA_PATH = Path(__file__).parent / 'data'
    X_train, train_labels = load_data(DATA_PATH / 'train.txt')
    X_test, test_labels = load_data(DATA_PATH / 'test.txt')

    # binarize target labels
    multi_label_binarizer = MultiLabelBinarizer()
    y_train = multi_label_binarizer.fit_transform(train_labels)
    y_test = multi_label_binarizer.transform(test_labels)
    labels = multi_label_binarizer.classes_

    # set initial seed
    SEED = 23249425
    random.seed(SEED)
    # set 10 seeds for randomization of models
    local_seeds = {run: random.randint(1, 2**32 - 1) for run in range(10)}

    # get best parameters for OvR Model
    ovr_best_params = None
    if not simple_model:
        ovr_best_params = ovr_hyperparameter_optimization(X_train, y_train, labels, SEED)

    # Benchmark
    scores_all_runs = []
    times_all_runs = []
    for run, seed in local_seeds.items():
        # build OvR model
        ovr_f1, ovr_accuracy, ovr_time = ovr_model(
            X_train, X_test, y_train, y_test, labels, seed,
            params=ovr_best_params)
        # build RAKEL model
        rakel_f1, rakel_accuracy, rakel_time = rakel_model(
            X_train, X_test, y_train, y_test, labels, seed)
        # capture scores in DataFrame
        scores_run = pd.DataFrame({
            'Emotion': labels,
            'run': run + 1,
            'ovr_f1': ovr_f1,
            'ovr_accuracy': ovr_accuracy,
            'rakel_f1': rakel_f1,
            'rakel_accuracy': rakel_accuracy,
        })
        scores_all_runs.append(scores_run)
        # capture times
        times_all_runs.append((run + 1, ovr_time, rakel_time))

    # extract relevant outputs
    scores = pd.concat(scores_all_runs)
    mean_scores, var_scores = calulate_results(scores)
    p_values = calculate_significance(scores, labels)
    times = pd.DataFrame.from_records(
        times_all_runs,
        columns=['run', 'ovr_sec', 'rakel_sec']
        ).round(3)
    seeds = pd.DataFrame.from_dict(
        local_seeds,
        orient='index',
        columns=['seed'])
    if not simple_model:
        ovr_best_params = pd.DataFrame.from_dict(ovr_best_params, orient='index')

    # write outputs to CSV and Latex files
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
    if not simple_model:
        ovr_best_params.to_csv(OUTPUT_PATH / 'ovr_best_params.csv', index_label='Emotion')
        ovr_best_params.to_latex(OUTPUT_PATH / 'ovr_best_params.txt')
