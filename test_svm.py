import warnings
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from bow import load_data_raw, create_bow
import numpy as np

warnings.filterwarnings("ignore")


DATA_FILES = Path(__file__).parent / 'data'
TEST_DATA = DATA_FILES / 'test.txt'
TRAIN_DATA = DATA_FILES / 'train.txt'

mlb = MultiLabelBinarizer()

train_data, train_target = load_data_raw(TRAIN_DATA)
test_data, test_target = load_data_raw(TEST_DATA)

# ndarray to remove deprecation warning
train_target = np.array(train_target, dtype='object')
test_target = np.array(test_target, dtype='object')

# merge datasets to get same length BOW vectors
X = np.concatenate([train_data, test_data], axis=0)
Y = np.concatenate([train_target, test_target], axis=0)

# training data with tfidf
X_bow = create_bow(X)

# binarize targets
Y_bin = mlb.fit_transform(Y)

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_bow, Y_bin, test_size=0.2, random_state=1337)

clf_l1 = OneVsRestClassifier(LinearSVC(penalty='l1', dual=False, max_iter=10e3))
clf_l2 = OneVsRestClassifier(LinearSVC(penalty='l2', random_state=1337, max_iter=10e3))
clfs = [clf_l1, clf_l2]
clf_l1.fit(X_train, y_train)

# Dry-run, just to test if it works
test_pred = clf_l1.predict(X_test)
pred_transform = mlb.inverse_transform(test_pred)
test_transform = mlb.inverse_transform(y_test)

for (test, pred) in zip(test_transform[:10], pred_transform[:10]):
    print(f'tags: {test}, predicted: {pred}')

# Grid search for hyperparams
parameters = dict(estimator__C=[0.1, 1, 10, 100, 1000])

print('\nTesting different penalties without tf-idf')
for clf in clfs:
    gs = GridSearchCV(clf, param_grid=parameters, refit=False, cv=5)
    gs.fit(X_train, y_train)
    scores = gs.cv_results_['mean_test_score']
    print(f'Penalty: {clf.estimator.penalty}, scores: {scores}')

# Testing with tf-idf
print('\nTesting different penalties with tf-idf')
X_bow_tf = create_bow(X, tfidf=True)
X_train, X_test, y_train, y_test = train_test_split(X_bow_tf, Y_bin, test_size=0.2, random_state=1337)

for clf in clfs:
    gs = GridSearchCV(clf, param_grid=parameters, refit=False, cv=5)
    gs.fit(X_train, y_train)
    scores = gs.cv_results_['mean_test_score']
    print(f'Penalty: {clf.estimator.penalty}, scores: {scores}')

# L2 penalty with tf-idf performs better so far
print('\nTesting log and tf-idf')
X_log = create_bow(X, tfidf=True, log=True)
X_train, X_test, y_train, y_test = train_test_split(X_log, Y_bin, test_size=0.2, random_state=1337)

# L1 with log tf-idf
gs = GridSearchCV(clf_l1, param_grid=parameters, refit=False, cv=5)
gs.fit(X_train, y_train)
print('L1 Log tf-idf, scores: {}'.format(gs.cv_results_['mean_test_score']))

# L2 with log tf-idf
gs = GridSearchCV(clf_l2, param_grid=parameters, refit=False, cv=5)
gs.fit(X_train, y_train)
print('L2 Log tf-idf, scores: {}'.format(gs.cv_results_['mean_test_score']))

# Oversampling (might not work)
# smote = SMOTE()
