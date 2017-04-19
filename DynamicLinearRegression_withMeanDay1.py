# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 23:24:06 2016

@author: ShivamMaurya
"""

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf as bpdf
#from LinearAndMultiLinearRegression import combineDay1Day2TestDay3
#from LinearAndMultiLinearRegression import combineDay2Day3TestDay1
#from LinearAndMultiLinearRegression import combineDay1Day3TestDay2


dataCompete = pd.ExcelFile('C:/MASTERS/Project/regression analysis/combinedData50000.xlsx') 
wsday1 =  dataCompete.parse('Sheet1')[0:50006]
wsday2 =  dataCompete.parse('Sheet2')[0:43168]
wsday3 =  dataCompete.parse('Sheet3')[0:43037]


#pdf file
pdf = bpdf.PdfPages("Analysis Plots - Dynamic Linear Regression_withMeanDay1.pdf")

for x in range(1,9):
    independent = np.array(wsday1[["input_heat_energy"+str(x),"input_wall_temp"+str(x),"input_mat_temp"+str(x)]])
    dependent = np.array(wsday1["output_section_temp"+str(x)])
    
    ##Initialize the model
    LinearRegressor = linear_model.LinearRegression()
    LinearRegressor.fit(independent[0:7200],dependent[0:7200])
    totalPredicted = np.array(LinearRegressor.predict(independent[0:7200]))
    staticPredict = LinearRegressor.predict(independent)
    
    plt.plot(wsday1['Time Stamp'], dependent, 'b-', label = 'Actual Data')
    plt.plot(wsday1['Time Stamp'], staticPredict , 'g--', label = 'Predicted Init Model')
    plt.plot(wsday1['Time Stamp'][0:7200], staticPredict[0:7200], 'r--', label = 'Dynamic Predicted')
    plt.xlabel('Time Stamp')
    plt.ylabel('Output Temperature'+str(x))
    plt.grid(True)
    plt.legend(loc='best',fontsize=7) 
    
    learningRate = 0.0000001
    
    for y in range(0,47):
        start = int(7200 + y*900)
        end = int(7200+(y+1)*900)
       
        
#        error = np.mean(np.array(predictDelta)) - np.mean(np.array(dependent[start:end]))
#    
#        LinearRegressor.intercept_ += -learningRate*error
#        LinearRegressor.coef_[0] += - learningRate*error
#        LinearRegressor.coef_[1] += - learningRate*error
#        LinearRegressor.coef_[2] += - learningRate*error

        error = np.mean(np.array(dependent[start:end])) - np.mean(np.array(LinearRegressor.predict(independent[start:end])))
        LinearRegressor.intercept_ += learningRate*error*1
        LinearRegressor.coef_[0] +=  learningRate*error*np.mean(independent[start:end,0:1])
        LinearRegressor.coef_[1] +=  learningRate*error*np.mean(independent[start:end,1:2])
        LinearRegressor.coef_[2] +=  learningRate*error*np.mean(independent[start:end,2:3])
        predictDelta = LinearRegressor.predict(independent[start:end])
        totalPredicted = np.append(totalPredicted,predictDelta,axis=0)
        plt.plot(wsday1['Time Stamp'][start:end],predictDelta,'r--')
    
#    print(error,totalPredicted.shape,dependent.shape)
    print(LinearRegressor.coef_,LinearRegressor.intercept_)
    print(r2_score(dependent[7200:49500],totalPredicted[7200:49500]))
    print(r2_score(dependent[7200:49500],staticPredict[7200:49500]))
    print()
    plt.savefig(pdf, format='pdf')
    plt.close()
    
pdf.close()
    
