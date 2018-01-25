#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-24 13:32:28
# @Author  : lebronran (lebronran@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import os
import numpy as np

# helper function
def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)
