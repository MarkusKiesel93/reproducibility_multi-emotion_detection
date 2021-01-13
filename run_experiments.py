import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind
from sklearn.metrics import f1_score, jaccard_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.ensemble import RakelD
from bow import BOW


# Run Exkeriments
def benchmark(bow, ovr, rakel, num_runs=10):
    results_all_runs = []

    X_train, X_test, y_train, y_test = bow.create()
    labels = bow.get_labels()

    for n in range(num_runs):
        # run OvR model
        ovr.fit(X_train, y_train)
        ovr_prediction = ovr.predict(X_test)
        ovr_f1 = f1_score(y_test, ovr_prediction, average=None)
        ovr_accuracy = jaccard_score(y_test, ovr_prediction, average=None)

        # run RAKEL model
        rakel.fit(X_train, y_train)
        rakel_prediction = rakel.predict(X_test)
        rakel_f1 = f1_score(y_test, rakel_prediction, average=None)
        rakel_accuracy = jaccard_score(y_test, rakel_prediction, average=None)

        # capture results in DataFrame
        results = pd.DataFrame({
            'Emotion': labels,
            'run': n,
            'ovr_f1': ovr_f1,
            'ovr_accuracy': ovr_accuracy,
            'rakel_f1': rakel_f1,
            'rakel_accuracy': rakel_accuracy,
        })
        results_all_runs.append(results)

    return pd.concat(results_all_runs), labels


# extract resulst from all runs
def calulate_results(scores, labels):
    grouped = scores.groupby('Emotion').agg(
        ovr_f1_mean=('ovr_f1', 'mean'),
        ovr_f1_var=('ovr_f1', 'var'),
        ovr_accuracy_mean=('ovr_accuracy', 'mean'),
        ovr_accuracy_var=('ovr_accuracy', 'var'),
        rakel_f1_mean=('rakel_f1', 'mean'),
        rakel_f1_var=('rakel_f1', 'var'),
        rakel_accuracy_mean=('rakel_accuracy', 'mean'),
        rakel_accuracy_var=('rakel_accuracy', 'var'),
    )
    return grouped


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


if __name__ == '__main__':

    # BOW
    bow = BOW(
        stop_words=True,
        tfidf=True,
        log=True,
    )

    # One-vs-Rest Model
    ovr = OneVsRestClassifier(
        LinearSVC(penalty='l1', dual=False, max_iter=10e3))

    # RAKEL Model

    rakel = RakelD(
        base_classifier=LinearSVC(C=1),
        base_classifier_require_dense=[True, True],
        labelset_size=3
    )

    # run experiments
    results, labels = benchmark(bow, ovr, rakel)
    all_results = calulate_results(results, labels)
    p_values = calculate_significance(results, labels)

    print(all_results)
    print(p_values)

    # save results
    # RESULTS_PATH = Path(__file__).parent / 'results'
    # all_results.to_csv(RESULTS_PATH / 'results.csv')
    # all_results.to_latex(RESULTS_PATH / 'results.txt')
    # p_values.to_csv(RESULTS_PATH / 'p_values.csv')
    # p_values.to_latex(RESULTS_PATH / 'p_values.txt')
