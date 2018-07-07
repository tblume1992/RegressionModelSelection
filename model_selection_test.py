#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 15:08:39 2018

@author: tyler
"""
import lr_simulation as simulate
import RegressionModelSelection as RMS

simulation = simulate.simulateData(n_samples = 1000, n_features = 11, noise = 20, multicollinearity = 0, 
                                   mc_correlation = 0.9, n_informative = 5, random_state = None)

Xhat, yhat,coefhat = simulation.make_Data()


ms = RMS.ModelSelection(Xhat, yhat)
ms.min_aic()
ms.min_bic()
ms.max_adj_rsq()
ms.max_rsq()
ms.min_test_error(test_size = .33)

