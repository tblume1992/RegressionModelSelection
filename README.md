# RegressionModelSelection
lr_simmulation generates data to be fed into a linear regression.  It is built on sklearn's 'make_regression' script but adds the ability to specify the number of multicollinearity variables and their correlation.  

RegressionModelSelection has 4 functions which will iterate through all possible variable combinations and choose the model which performs best according to the metric that is chosen. 
