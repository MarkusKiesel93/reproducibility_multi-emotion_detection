import warnings
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from bow import BOW
import numpy as np

warnings.filterwarnings("ignore")

# load BOW features
bow_default = BOW()
X_train, X_test, y_train, y_test = bow_default.create()

clf_l1 = OneVsRestClassifier(LinearSVC(penalty='l1', dual=False, max_iter=10e3))
clf_l2 = OneVsRestClassifier(LinearSVC(penalty='l2', random_state=1337, max_iter=10e3))
clfs = [clf_l1, clf_l2]
clf_l1.fit(X_train, y_train)

# Dry-run, just to test if it works
test_pred = clf_l1.predict(X_test)
pred_transform = bow_default.get_original_labels(test_pred)
test_transform = bow_default.get_original_labels(y_test)

for (test, pred) in zip(test_transform[:10], pred_transform[:10]):
    print(f'tags: {test}, predicted: {pred}')

# Grid search for hyperparams
parameters = dict(estimator__C=[0.1, 1, 10, 100, 1000])

print('\nTesting different penalties without tf-idf')
for clf in clfs:
    gs = GridSearchCV(clf, verbose=3, param_grid=parameters, refit=False, cv=5, scoring='f1_micro')
    gs.fit(X_train, y_train)
    scores = gs.best_score_
    print(f'Penalty: {clf.estimator.penalty}, scores: {scores}')
    print('best params: {}'.format(gs.best_params_))

# Testing with tf-idf
print('\nTesting different penalties with tf-idf')
bow_tf = BOW(tfidf=True)
X_train, X_test, y_train, y_test = bow_default.create()

for clf in clfs:
    gs = GridSearchCV(clf, verbose=3, param_grid=parameters, refit=False, cv=5, scoring='f1_micro')
    gs.fit(X_train, y_train)
    scores = gs.best_score_
    print(f'Penalty: {clf.estimator.penalty}, scores: {scores}')
    print('best params: {}'.format(gs.best_params_))

# L2 penalty with tf-idf performs better so far
print('\nTesting log and tf-idf')
bow_log = BOW(tfidf=True, log=True)
X_train, X_test, y_train, y_test = bow_default.create()

# L1 with log tf-idf
gs = GridSearchCV(clf_l1, verbose=3, param_grid=parameters, refit=False, cv=5, scoring='f1_micro')
gs.fit(X_train, y_train)
print('L1 Log tf-idf, scores: {}'.format(gs.best_score_))
print('best params: {}'.format(gs.best_params_))

# L2 with log tf-idf
gs = GridSearchCV(clf_l2, verbose=3, param_grid=parameters, refit=False, cv=5, scoring='f1_micro')
gs.fit(X_train, y_train)
print('L2 Log tf-idf, scores: {}'.format(gs.best_score_))
print('best params: {}'.format(gs.best_params_))
# Oversampling (might not work)
# smote = SMOTE()
