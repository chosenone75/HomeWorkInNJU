#!/usr/bin/env python  
# encoding: utf-8    
""" 
@version: v1.0 
@author: lebronran 
@contact: lebronran@gmail.com
@file: linear_model.py
@time: 17-11-13 下午1:41 
"""
# import
from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import dataloader
import  numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)

# filepath
DATASET = "dataset"
TRAIN = "train"
DEV = "dev"
TEST = "test"

# load dataset
train_X, train_Y = dataloader.load_files(DATASET,TRAIN,shuffle=False,random_state=42)
dev_X, dev_Y = dataloader.load_files(DATASET,DEV,shuffle=False,random_state=42)
test_X = dataloader.load_test_files(DATASET,TEST)

# build a pipeline
clf = Pipeline([
    ('vect',CountVectorizer(min_df=1)),
    ('tfidf',TfidfTransformer()),
    #('lr',LogisticRegression(solver='newton-cg'))
    ('svm', SGDClassifier())
])

# grid search
parameters = {
    'vect__ngram_range': [(1, 3), (1, 4)],
    'tfidf__use_idf': (True, False),
    #'lr__multi_class': ('multinomial', 'ovr'),
    #'lr__C':(0.7,0.85,1),
    'svm__alpha':(0.01,0.0105,0.011),
}
print("Starting Grid Search...")
beign = time.time()
gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
gs_clf.fit(dev_X,dev_Y)
end = time.time()
logging.info("Time Cost: %s" % (end - beign))

print(gs_clf.best_score_)
for name in sorted(parameters.keys()):
    print("%s:%r" % (name, gs_clf.best_params_[name]))

clf_best = gs_clf.best_estimator_
clf_best.fit(np.concatenate((train_X, dev_X),axis=0),np.concatenate((train_Y,dev_Y),axis=0))

# check performance
train_prediction = clf_best.predict(train_X)
print("train set correct rate: %f" % np.mean(train_prediction == train_Y))

dev_prediction = clf_best.predict(dev_X)
print("dev set correct rate: %f" % np.mean(dev_prediction == dev_Y))

# test
test_prediction = clf_best.predict(test_X)
# save to file
np.savetxt('output/demo.txt',test_prediction,fmt="%d",delimiter="\n")