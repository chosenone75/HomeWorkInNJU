from collections import defaultdict
import numpy as np
import re
import numbers

# misc function


def clean_str(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def load_files(filepath, subset, shuffle=False, random_state=None):
    with open(filepath + "/" + subset + "_x.txt") as file_X,\
            open(filepath + "/" + subset + "_y.txt") as file_Y:
        data_X = file_X.read().strip().lower().split('\n')
        data_Y = file_Y.read().strip().lower().split('\n')
        data_X = [clean_str(text) for text in data_X]
        data_X = np.array(data_X)
        label_Y = np.asarray(data_Y, dtype=np.int)
        data_Y = np.zeros(shape=(len(data_Y), 5))
        data_Y[range(len(label_Y)), label_Y - 1] = 1
        if shuffle:
            random_stat = check_random_state(random_state)
            indics = np.arange(data_X.shape[0])
            random_stat.shuffle(indics)
            data_X = data_X[indics]
            data_Y = data_Y[indics]
        return data_X, data_Y


def load_test_files(filepath, subset):
    with open(filepath + "/" + subset + "_x.txt") as file_X:
        test_X = file_X.read().strip().lower().split('\n')
        test_X = [clean_str(text) for text in test_X]
        test_X = np.array(test_X)
        return test_X


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def build_data_vocab(file_path):
    vocab = defaultdict(float)

    with open(file_path) as f:
        for line in f:
            sent = clean_str(line)
            words = set(re.split('\s',sent))
            for word in words:
                vocab[word] += 1
        return vocab

def get_WordVector(word_vecs,vocab, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    W = np.zeros(shape=(vocab_size, k), dtype='float32')

    for idx, word in enumerate(vocab):
        W[idx] = word_vecs[word]
    return W

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, k=300):
    """
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

if __name__ == '__main__':
    train_x, train_y = load_files('data', 'train')
    print(train_x.shape, train_y.shape)
    print(train_y[:5])

    print(len(build_data_vocab('data/train_x.txt')))