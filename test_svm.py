import warnings
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer
from bow import BOW
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# load BOW features
bow_default = BOW()
X_train, X_test, y_train, y_test = bow_default.create()

clf_l1 = OneVsRestClassifier(LinearSVC(penalty='l1', dual=False, max_iter=10e3, class_weight='balanced',
                                       multi_class='ovr', verbose=0))
clf_l1.fit(X_train, y_train)

# Dry-run, just to test if it works
test_pred = clf_l1.predict(X_test)
pred_transform = bow_default.get_original_labels(test_pred)
from sklearn.pipeline import Pipeline
test_transform = bow_default.get_original_labels(y_test)

for (test, pred) in zip(test_transform[:10], pred_transform[:10]):
	print(f'tags: {test}, predicted: {pred}')



def train_models(X_train, y_train, clf_l1):
    # Grid search for hyperparams
    parameters = {'estimator__C':[0.1, 1, 10, 100, 1000], 'estimator__penalty':['l1','l2']}
    best_params = {}
    for c in parameters['estimator__C']:
        for pen in parameters['estimator__penalty']:
            clf_l1.set_params(estimator__C=c, estimator__penalty=pen)
            clf_l1.fit(X_train, y_train)
            score = []
            for i in range(len(clf_l1.estimators_)):
                pred = clf_l1.estimators_[i].predict(X_test)
                cv_score = np.mean(cross_val_score(clf_l1.estimators_[i], X_train, y_train[:,i], cv=5,
                                           scoring=make_scorer(f1_score, average='micro')))
                if str(i) in best_params.keys():
                    if best_params[str(i)]['score'] < cv_score:
                        best_params[str(i)]['score'] = cv_score
                        best_params[str(i)]['params'] = f'C={c}, Penalty={pen}'
                else:
                    best_params[str(i)] = {'score':0, 'params':None}
                    best_params[str(i)]['score'] = cv_score
                    best_params[str(i)]['params'] = f'C={c}, Penalty={pen}'
    return best_params

# Testing without tf-idf
print('Testing without tf-idf')
best_params_notf = train_models(X_train, y_train, clf_l1)

# Testing with tf-idf
print('Testing with tf-idf')
bow_tf = BOW(tfidf=True)
X_train, X_test, y_train, y_test = bow_default.create()
best_params_tfidf = train_models(X_train, y_train, clf_l1)



# Testing with tf-idf and log
print('Testing with tf-idf and log')
bow_log = BOW(tfidf=True, log=True)
X_train, X_test, y_train, y_test = bow_default.create()
best_params_tf_log = train_models(X_train, y_train, clf_l1)


df_notf = pd.DataFrame.from_dict(best_params_notf, orient='index')
df_tfidf = pd.DataFrame.from_dict(best_params_tfidf, orient='index')
df_tflog = pd.DataFrame.from_dict(best_params_tf_log, orient='index')
print(df_notf)
print(df_tfidf)
print(df_tflog)

"""
print('\nTesting different penalties without tf-idf')
for clf in clfs:
	gs = GridSearchCV(clf, verbose=4, param_grid=parameters, refit=True, cv=5, scoring='f1_samples')
	gs.fit(X_train, y_train)
	y_true, y_pred = y_test, gs.predict(X_test)
	means = gs.cv_results_['mean_test_score']
	stds = gs.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	print(classification_report(y_true, y_pred))
	scores = gs.best_score_
	print(f'Penalty: {clf.estimator.penalty}, scores: {scores}')
	print('best params: {}'.format(gs.best_params_))

# Testing with tf-idf
print('\nTesting different penalties with tf-idf')
bow_tf = BOW(tfidf=True)
X_train, X_test, y_train, y_test = bow_default.create()

for clf in clfs:
	gs = GridSearchCV(clf, verbose=3, param_grid=parameters, refit=False, cv=5, scoring='f1_macro')
	gs.fit(X_train, y_train)
	scores = gs.best_score_
	print(f'Penalty: {clf.estimator.penalty}, scores: {scores}')
	print('best params: {}'.format(gs.best_params_))

# L2 penalty with tf-idf performs better so far
print('\nTesting log and tf-idf')
bow_log = BOW(tfidf=True, log=True)
X_train, X_test, y_train, y_test = bow_default.create()

# L1 with log tf-idf
gs = GridSearchCV(clf_l1, verbose=3, param_grid=parameters, refit=False, cv=5, scoring='f1_macro')
gs.fit(X_train, y_train)
print('L1 Log tf-idf, scores: {}'.format(gs.best_score_))
print('best params: {}'.format(gs.best_params_))

# L2 with log tf-idf
gs = GridSearchCV(clf_l2, verbose=3, param_grid=parameters, refit=False, cv=5, scoring='f1_macro')
gs.fit(X_train, y_train)
print('L2 Log tf-idf, scores: {}'.format(gs.best_score_))
print('best params: {}'.format(gs.best_params_))
# Oversampling (might not work)
"""
