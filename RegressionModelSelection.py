#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:42:16 2018

@author: tyler
"""
import statsmodels.api as sm
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
class ModelSelection():
    def __init__(self, X, y):
        self.X =pd.DataFrame(X)
        self.y = y
        
    def min_aic(self):
        aic = 10**100
        aic_dataset = []
        for L in range(0, len(self.X.columns.values)+1):
            for subset in itertools.combinations(self.X.columns.values, L):
                if len(subset) < 1:
                    pass
                else:
                    mod = sm.GLM(self.y, self.X.get(list(subset)))
                    res = mod.fit()
                    
                    if aic > sm.regression.linear_model.RegressionResults.aic(res):
                        aic = sm.regression.linear_model.RegressionResults.aic(res)
                        aic_dataset.append(aic)
                        model = sm.GLM(self.y, self.X.get(list(subset)))
        
        res = model.fit()
        print("AIC Minimization")
        print(res.summary())
        
    def min_bic(self):
        bic = 10**100
        bic_dataset = []
        for L in range(0, len(self.X.columns.values)+1):
            for subset in itertools.combinations(self.X.columns.values, L):
                if len(subset) < 1:
                    pass
                else:
                    mod = sm.GLM(self.y, self.X.get(list(subset)))
                    res = mod.fit()
                    
                    if bic > sm.regression.linear_model.RegressionResults.bic(res):
                        bic = sm.regression.linear_model.RegressionResults.bic(res)
                        bic_dataset.append(bic)
                        model = sm.GLM(self.y, self.X.get(list(subset)))
        
        res = model.fit()
        print("BIC Minimization")
        print(res.summary())

    def max_rsq(self):
        rsq = .001
        for L in range(0, len(self.X.columns.values)+1):
            for subset in itertools.combinations(self.X.columns.values, L):
                if len(subset) < 1:
                    pass
                else:
                    mod = sm.OLS(self.y, self.X.get(list(subset)))
                    res = mod.fit()
                    
                    if rsq < sm.regression.linear_model.RegressionResults.rsquared(res):
                        rsq = sm.regression.linear_model.RegressionResults.rsquared(res)
                        model = sm.OLS(self.y, self.X.get(list(subset)))
        res = model.fit()
        print("R-Squared Maximization")
        print(res.summary())
    
    def max_adj_rsq(self):
        adj_rsq = .001
        for L in range(0, len(self.X.columns.values)+1):
            for subset in itertools.combinations(self.X.columns.values, L):
                if len(subset) < 1:
                    pass
                else:
                    mod = sm.OLS(self.y, self.X.get(list(subset)))
                    res = mod.fit()
                    
                    if adj_rsq < sm.regression.linear_model.RegressionResults.rsquared_adj(res):
                        adj_rsq = sm.regression.linear_model.RegressionResults.rsquared_adj(res)
                        model = sm.OLS(self.y, self.X.get(list(subset)))
        res = model.fit()
        print("Adj R-Squared Maximization")
        print(res.summary())
    
    def min_test_error(self, test_size = .33, random_state = None):
        mse_list = []
        mse = 10**100
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size= test_size, random_state=random_state)
        for L in range(0, len(X_train.columns.values)+1):
            for subset in itertools.combinations(X_train.columns.values, L):
                if len(subset) < 1:
                    pass
                else:
                    mod = sm.GLM(y_train, X_train.get(list(subset)))
                    res = mod.fit()
                    y_pred = res.predict(X_test.get(list(subset)))
                    mse_list.append(mean_squared_error(y_test, y_pred))
                    if mse > mean_squared_error(y_test, y_pred):
                        mse =  mean_squared_error(y_test, y_pred)
                        split_model = sm.GLM(self.y, self.X.get(list(subset)))
        split_res = split_model.fit()
        print("Min Test MSE")
        print(split_res.summary())
