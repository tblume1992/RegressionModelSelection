#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 09:49:53 2018

@author: tyler
"""
import numpy as np
from sklearn.datasets import make_regression


# Generate toy data.
class simulateData():
    def __init__(self, n_samples=500, n_features=10, n_informative = 5,
                 random_state=None, noise=10.0, bias=100.0, coef = True,
                 multicollinearity = 0, mc_correlation = .95
                 ):
        self.n_samples= n_samples
        self.n_features= n_features - multicollinearity
        self.n_informative = n_informative
        self.random_state= random_state
        self.noise= noise
        self.bias= bias
        self.coef = coef
        self.multicollinearity = multicollinearity
        self.mc_correlation = mc_correlation
    def select_mcVars(self, X, variables):
        nonzero_coef = np.nonzero(variables)[0]
        mcIndex = nonzero_coef[:self.multicollinearity]
        return X[:,mcIndex]
    def make_Data(self):    
        X, y, coef = make_regression(n_samples=self.n_samples, n_features=self.n_features,
                                     n_informative = self.n_informative, random_state=self.random_state,
                                     noise=self.noise, bias=self.bias, coef = self.coef)
        if self.multicollinearity > 0:
            mcVars = self.select_mcVars(X, coef)
            correlation_matrix = np.tile(self.mc_correlation, len(mcVars.T))
            MC = np.dot(mcVars,np.linalg.cholesky(np.diag(correlation_matrix)))
            X = np.column_stack((X,MC))
        return X, y, coef



        