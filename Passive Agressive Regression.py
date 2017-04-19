# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 23:42:42 2016

@author: ShivamMaurya
"""
import pandas
import numpy as np
#from sklearn import svm 
from sklearn import linear_model
#from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

##############################################################################
# DATA STAGING
##############################################################################
dataCompete = pandas.ExcelFile('Combined Data.xlsx') 
ws =  dataCompete.parse('Sheet1')
target = np.array(ws['outputTemp16'])
#target = target.reshape(1,1)
independentVariable = (np.array(ws[['inputWall16','inputEnergy16']]))
independentVariable = independentVariable.reshape(50400,2)

targetTrain = target[0:37800]
independentVariableTrain = independentVariable[0:37800]

independentVariableTrain.reshape(37800,2)

targetTest = target[37800:50400]
independentVariableTest = (independentVariable[37800:50400])

independentVariableTest.reshape(12600,2)

# %% Passive Aggressive Regression

regrPAR = linear_model.PassiveAggressiveRegressor(loss='epsilon_insensitive')
regrPAR.fit(independentVariableTrain,targetTrain)

regrOutPAR = linear_model.PassiveAggressiveRegressor(loss='epsilon_insensitive')
regrOutPAR.fit(independentVariableTest,targetTest)


predictedValidatePAR = regrPAR.predict(independentVariableTest)
# Predict from the training features only
predictedTrainPAR = regrPAR.predict(independentVariableTrain)
# Predict from the whole training set (training+validation features)
predictedPAR = regrPAR.predict(independentVariable)

# Compute the root mean squared (RMS) for each case above
RMSvalid = sqrt(mean_squared_error(targetTest, predictedValidatePAR))
RMStrain = sqrt(mean_squared_error(targetTrain, predictedTrainPAR))
RMS = sqrt(mean_squared_error(target, predictedPAR))
# Print to standard IO the RMS and prediction scores.

print('Passive Aggressive Regression: RMS on validation set = ', RMSvalid)
print('Passive Aggressive Regression: RMS on training set = ', RMStrain)
print('Passive Aggressive Regression: RMS on whole set = ', RMS)

print('Passive Aggressive Regression: Score on validation set = ', regrPAR.score(independentVariableTest, targetTest))
print('Passive Aggressive Regression: Score on training set = ', regrPAR.score(independentVariableTrain, targetTrain))
print('Passive Aggressive Regression: Score on whole set = ', regrPAR.score(independentVariable, target))

# %% Linear Regression
regrLR = linear_model.LinearRegression()
regrLR.fit(independentVariableTrain,targetTrain)

#regrOutLR = linear_model.LinearRegression()
#regrOutLR.fit(independentVariableTest,targetTest)

predictedValidateLR = regrLR.predict(independentVariableTest)
# Predict from the training features only
predictedTrainLR = regrLR.predict(independentVariableTrain)
# Predict from the whole training set (training+validation features)
predictedLR = regrLR.predict(independentVariable)

# Compute the root mean squared (RMS) for each case above
RMSvalid = sqrt(mean_squared_error(targetTest, predictedValidateLR))
RMStrain = sqrt(mean_squared_error(targetTrain, predictedTrainLR))
RMS = sqrt(mean_squared_error(target, predictedLR))

# Print to standard IO the RMS and prediction scores.

print('Linear Regression: RMS on validation set = ', RMSvalid)
print('Linear Regression: RMS on training set = ', RMStrain)
print('Linear Regression: RMS on whole set = ', RMS)

print('Linear Regression: Score on validation set = ', regrLR.score(independentVariableTest, targetTest))
print('Linear Regression: Score on training set = ', regrLR.score(independentVariableTrain, targetTrain))
print('Linear Regression: Score on whole set = ', regrLR.score(independentVariable, target))


plt.plot(ws['time'][37800:50400], targetTest, 'b-')
plt.plot(ws['time'][37800:50400], predictedValidatePAR, 'r--')
plt.plot(ws['time'][37800:50400], predictedValidateLR, 'g--')
plt.xlabel('Time')
plt.ylabel('Output Temperature')
plt.show()
