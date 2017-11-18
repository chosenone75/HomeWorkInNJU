#!/usr/bin/env python  
# encoding: utf-8    
""" 
@version: v1.0 
@author: lebronran 
@contact: lebronran@gmail.com
@file: dataloader.py 
@time: 17-11-13 下午1:41 
"""
from __future__ import print_function
import numpy as np
import numbers

# misc function
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
    with open(filepath+"/"+subset+"_x.txt") as file_X,\
        open(filepath+"/"+subset+"_y.txt") as file_Y:
        data_X = file_X.read().strip().lower().split('\n')
        data_Y = file_Y.read().strip().lower().split('\n')
        data_X = np.array(data_X)
        data_Y = np.asarray(data_Y, dtype=np.float)
        if shuffle:
            random_stat = check_random_state(random_state)
            indics = np.arange(data_X.shape[0])
            random_stat.shuffle(indics)
            data_X = data_X[indics]
            data_Y = data_Y[indics]

        return data_X,data_Y

def load_test_files(filepath, subset):
    with open(filepath+"/"+subset+"_x.txt") as file_X:
        test_X = file_X.read().strip().lower().split('\n')
        test_X = np.array(test_X)
        return test_X

if __name__ == "__main__":
    data_X,data_Y = load_files("dataset",'dev',shuffle=True,random_state=42)
    print(len(data_X),len(data_Y))
    print(data_X.shape,data_Y.shape)
    print(data_X[0])
    print(data_Y[0])