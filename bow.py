from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# todo: stop words 
# - not clear if used for all as RAKEL uses log tfid

# todo: try other splits
# - combine train and test
# - use train_test_split for new split
# - allow setting seed

DATA_FILES = Path(__file__).parent / 'data'
TEST_DATA = DATA_FILES / 'test.txt'
TRAIN_DATA = DATA_FILES / 'train.txt'

class BOW:
    def __init__(self, stop_words=False, tfidf=False, log=False):
        self.tfidf = tfidf
        sw = None
        if stop_words:
            sw = 'english'
        self.count_vectorizer = CountVectorizer(stop_words=sw)
        self.tfid_transformer = TfidfTransformer(sublinear_tf=log)
        self.multi_label_binarizer = MultiLabelBinarizer()


    def create(self):
        # load raw data
        train_data, train_target = self.load_data_raw(TRAIN_DATA)
        test_data, test_target = self.load_data_raw(TEST_DATA)

        # create BOW features
        pipe_inp = [('bow', self.count_vectorizer)]
        if self.tfidf:
            pipe_inp.append(('tfidf', self.tfid_transformer))
        pipe = Pipeline(pipe_inp)
        X_train = pipe.fit_transform(train_data)
        X_test = pipe.transform(test_data)

        # binarize target
        y_train = self.multi_label_binarizer.fit_transform(train_target)
        y_test = self.multi_label_binarizer.transform(test_target)

        return X_train, X_test, y_train, y_test

    def get_original_labels(self, y):
        return self.multi_label_binarizer.inverse_transform(y)

    def get_labels(self):
        return self.multi_label_binarizer.classes_

    # load raw data
    def load_data_raw(self, file_path):
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
