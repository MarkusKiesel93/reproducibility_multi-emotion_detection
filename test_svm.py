from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from bow import load_data_raw, create_bow


DATA_FILES = Path(__file__).parent / 'data'
TEST_DATA = DATA_FILES / 'test.txt'
TRAIN_DATA = DATA_FILES / 'train.txt'

mlb = MultiLabelBinarizer()

train_data, train_target = load_data_raw(TRAIN_DATA)
# training data with tfidf
train_bow = create_bow(train_data)

# training data without tfidf
train_bow_notf = create_bow(train_data, tfidf=False)

train_target_binarized = mlb.fit_transform(train_target)
print(train_target_binarized[:10])

print(train_bow.toarray()[:10])
print(train_bow_notf.toarray()[:10])
