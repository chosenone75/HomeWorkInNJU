#!/usr/bin/env python  
# encoding: utf-8    
""" 
@version: v1.0 
@author: lebronran 
@contact: lebronran@gmail.com
@file: demo.py 
@time: 17-11-12 下午9:37 
"""
from __future__ import print_function

from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import  fetch_20newsgroups
import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


logging.basicConfig(level=logging.INFO)

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',categories=categories,
                                  shuffle=True,random_state=42)
# check datatset
logging.info(twenty_train.target_names)
logging.info(len(twenty_train.data))
logging.info(len(twenty_train.filenames))

# check content of dataset
logging.info("\n".join(twenty_train.data[0].split("\n")[:3]))
logging.info(twenty_train.target_names[twenty_train.target[0]])
logging.info(twenty_train.target[:10])
logging.info(type(twenty_train.data))

# feature extraction bag of word
cout_vect = CountVectorizer()
X_train_counts = cout_vect.fit_transform(twenty_train.data)
print(type(X_train_counts),X_train_counts.shape)
logging.info(cout_vect.vocabulary_.get(u'algorithm'))

## tf-idf extractor
tf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)


# train a baseline classifier
clf = MultinomialNB().fit(X_train_tfidf,twenty_train.target)


# test on a fake 'new' doc

doc_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = cout_vect.transform(doc_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)
prediction = clf.predict(X_new_tfidf)
for doc,category in zip(doc_new,prediction):
    print("%r --> %s" % (doc,twenty_train.target_names[category]))

# build a pipeline
text_clf = Pipeline([('vect',CountVectorizer()),
                     ('tfidf',TfidfTransformer(use_idf=True)),
                     ('nb_clf',MultinomialNB()),
                     ])
text_clf_svm = Pipeline([('vect',CountVectorizer()),
                     ('tfidf',TfidfTransformer(use_idf=True)),
                     ('svm_clf',SGDClassifier('hinge', 'l2', alpha=1e-3, random_state=42)),
                     ])
text_clf_svm.fit(twenty_train.data,twenty_train.target)


# evaluation performance on the test set
twenty_test = fetch_20newsgroups(subset='test',categories=categories,\
                                 shuffle=True,random_state=42)
logging.info(len(twenty_test))

doc_test = twenty_test.data
test_prediction = text_clf_svm.predict(doc_test)

print("correct rate: %f" % np.mean(test_prediction == twenty_test.target))

print(metrics.classification_report(twenty_test.target,test_prediction,
                                    target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, test_prediction))

# grid search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'svm_clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf_svm, parameters, n_jobs=-1)
gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
print(gs_clf.best_score_)
for name in sorted(parameters.keys()):
    print("%s:%r" % (name, gs_clf.best_params_[name]))

clf_best = gs_clf.best_estimator_
clf_best.fit(twenty_train.data,twenty_train.target)

test_prediction = clf_best.predict(doc_test)

print("correct rate: %f" % np.mean(test_prediction == twenty_test.target))





















