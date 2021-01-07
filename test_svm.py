from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from bow import load_data_raw, create_bow
import numpy as np

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

# training data without tfidf
X_bow_tf = create_bow(X, tfidf=False)

# binarize targets
Y_bin = mlb.fit_transform(Y)

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_bow, Y_bin, test_size=0.2, random_state=1337)

clf = OneVsRestClassifier(LinearSVC())
clf.fit(X_train, y_train)

# prepare test data, length are not same! padding?
# padding won't work, the word indices are not the same
# solution: merge the two datasets and do train-test split?
test_pred = clf.predict(X_test)
pred_transform = mlb.inverse_transform(test_pred)
test_transform = mlb.inverse_transform(y_test)

for (test, pred) in zip(test_transform[:10], pred_transform[:10]):
    print(f'tags: {test}, predicted: {pred}')

