from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

DATA_FILES = Path(__file__).parent / 'data'
TEST_DATA = DATA_FILES / 'test.txt'
TRAIN_DATA = DATA_FILES / 'train.txt'


# load raw data
def load_data_raw(file_path):
    X = []
    y = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        sentence = ' '.join(line.split()[0:-1])
        X.append(sentence)
        labels = line.split()[-1].split('_')
        y.append(labels)
    return X, y


# create bow features
def create_bow(sentences, tfidf=False, log=False):
    pipe_inp = [('bow', CountVectorizer())]
    if tfidf:
        pipe_inp.append(('tfidf', TfidfTransformer(sublinear_tf=log)))
    pipe = Pipeline(pipe_inp)
    bow = pipe.fit_transform(sentences)
    return bow

if __name__ == '__main__':

    # test loading raw data
    X_train, y_train = load_data_raw(TRAIN_DATA)
    X_test, y_test = load_data_raw(TEST_DATA)
    print(f'X_train length: {len(X_train)}')
    print(f'y_train length: {len(y_train)}')
    print(f'X_train length: {len(X_test)}')
    print(f'X_train length: {len(y_test)}')
