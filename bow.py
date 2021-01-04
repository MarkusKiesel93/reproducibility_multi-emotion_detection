from pathlib import Path

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
def create_bow(sentences):
    # todo: implement
    pass


if __name__ == '__main__':

    # test loading raw data
    X_train, y_train = load_data_raw(TRAIN_DATA)
    X_test, y_test = load_data_raw(TEST_DATA)
    print(f'X_train length: {len(X_train)}')
    print(f'y_train length: {len(y_train)}')
    print(f'X_train length: {len(X_test)}')
    print(f'X_train length: {len(y_test)}')