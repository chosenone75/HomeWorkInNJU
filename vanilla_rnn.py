from datetime import datetime
import numpy as np
import logging
import csv
import itertools
import nltk
import operator
import sys
import matplotlib.pyplot as plt
import util

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the training data and append the SENTENCE_START and SENTENCE_END tokens
print("Reading Training Data...")
with open('data/reddit-comments-2015-08.csv', 'rt', encoding='utf8') as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    # split the comments into sentences
    sentences = itertools.chain(
        *[nltk.sent_tokenize(x[0].lower()) for x in reader])
    sentences = ["%s %s %s" %
                 (sentence_start_token, x, sentence_end_token) for x in sentences]

print("Parsed %d sentences" % len(sentences))
# print(sentences[:5])
tokenized_sentences = [nltk.word_tokenize(x) for x in sentences]
# print(tokenized_sentences[:5])

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Total tokens: %d." % (len(word_freq.items())))

# build the one-hot vector
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Vocablary size: %d" % vocabulary_size)
print("the least frequent word is %s and appeared %d times" %
      (vocab[-1][0], vocab[-1][1]))

for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [
        w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: %s" % sentences[0])
print("\nExample sentence after preprocessing: %s" % tokenized_sentences[0])


# create the training dataset
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]]
                      for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]]
                      for sent in tokenized_sentences])


class BasicRNN(object):
    """docstring for BasicRNN"""

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        super(BasicRNN, self).__init__()
        # assign instance vatiables
        # e.g self.arg = arg
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # randomly initialize the networks weights
        self.U = np.random.uniform(-np.sqrt(1.0/word_dim),
                                   np.sqrt(1.0/word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1.0/hidden_dim),
                                   np.sqrt(1.0/hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1.0/hidden_dim),
                                   np.sqrt(1.0/hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        s = np.zeros((T+1, self.hidden_dim))
        o = np.zeros((T, self.word_dim))
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)

        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1

        # do backwards with respect tp  each time step's output
        # see the details of bptt in "NOTEs of RNN"
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,[x[bptt_step]]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - (s[bptt_step-1] ** 2))
        return [dLdU, dLdV, dLdW]

    # a powerful tool to check whether the implemention of backpropagation process is correct.
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_gradients = self.bptt(x,y)
        model_parameters = ['U', 'V', 'W']
        for pidx, pname in enumerate(model_parameters):
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d" % (pname, np.prod(parameter.shape)))
            # iterate over each element of the parameter matrix, e.g. (0,0),(0,1),...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = parameter[ix]
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus) / (2*h)
                parameter[ix] = original_value
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate the relative error: (|x - y|/(|x|+|y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                if relative_error > error_threshold:
                    print("Gradient check ERROR: parameter=%s, ix=%s" % (pname, ix))
                    print("+h loss: %f" % gradplus)
                    print("-h loss: %f" & gradminus)
                    print("Estimated gradient: %f" % estimated_gradient)
                    print("backpropagation gradient: %f" % backprop_gradient)
                    print("relative error: %f" % relative_error)
                    return
                it.iternext()
            print("Gradient check pass for parameter %s" % pname)

