# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 18:12:26 2020

@author: XXXXXXXX
"""
# Importing the libraries #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing a dataset#
dataset = pd.read_csv('Salary_Data.csv') 

#Creating a matrix of Independent varialbe#
X= dataset.iloc[:, :-1].values  #Change#
print(X)

#Creating a matrix of Dependent varialbe#
Y= dataset.iloc[:, -1].values   #Change#
print(Y)

# SPLITTING THE TEST AND TRAINING DATASET#
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 1)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# FEATURE SCALING#
#Different ways of feature scaling: Standardisation and Normalization#
"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) #in training set, it is importance to do fit and then transform; unlike Test dataset#
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)"""
#Note: In simple linear regression, we do not need to use Feature scaling#

#Simple linear regression training on the Training dataset#
from sklearn.linear_model import LinearRegression #LinearRegression is the class which is inbuilt in the linear_model library in the Scikit-learn package#
#Create a regressor object to fit a model#
regressor = LinearRegression()
#fit method is used to fit the training dataset into the regressor object#
regressor.fit(X_train, Y_train)

#Print the constant and the coefficient# 
print(f'constant = {regressor.intercept_}')
print(f'coefficient = {regressor.coef_}')

#Predict method is used to do prediction on test dataset using regressor object# It predict how salary increases based on the experiences#
#Y_pred is the vector of the predicted salary#
Y_pred= regressor.predict(X_test)

#Visualising the training algorithm output#
#According to the plot the red dots are real values which is used in the training dataset, whereas blue line is the predicted values#
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')  
plt.xlabel('Years of Experience')                 
plt.ylabel('Salary')                             
plt.show()

#Visualizing the test dataset#
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')     
plt.xlabel('Years of Experience')                
plt.ylabel('Salary')                             
plt.show()

#Model evaluation#
from sklearn import metrics
r_square = metrics.r2_score(Y_test, Y_pred)
print('R-Square Error:', r_square)
