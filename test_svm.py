from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from bow import load_data_raw, create_bow


DATA_FILES = Path(__file__).parent / 'data'
TEST_DATA = DATA_FILES / 'test.txt'
TRAIN_DATA = DATA_FILES / 'train.txt'

train_data, train_target = load_data_raw(TRAIN_DATA)
# training data with tfidf
train_bow = create_bow(train_data)

# training data without tfidf
train_bow_notf = create_bow(train_data, tfidf=False)
print(train_bow[:10])
print(train_bow_notf[:10])
