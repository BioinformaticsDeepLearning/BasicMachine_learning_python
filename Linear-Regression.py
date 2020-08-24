# Importing the libraries #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing a dataset#
dataset = pd.read_csv('Data.csv')

#Creating a matrix of Independent varialbe#
X= dataset.iloc[:, :-1].values
print(X)

#Creating a matrix of Dependent varialbe#
Y= dataset.iloc[:, -1].values
print(Y)

# SPLITTING THE TEST AND TRAINING DATASET#
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

#Note: If your dataset is cleaned with any error or noise then, user can skip Data preprocessing step also#

#Simple linear regression training on the Training dataset#
from sklearn.linear_model import LinearRegression

#Create a regressor to fit a model#
reg = LinearRegression()

#reg is using to fit train dataset#
reg.fit(X_train, Y_train)

#Predicting the test results on the test dataset#
Y_pred= reg.predict(X_test)

#Visualising the training algorithm output#
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title('Title (Training set)')
plt.xlabel('Name of the X bar')
plt.ylabel('Name of the Y bar')
plt.show()

#Visualizing the test dataset#
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Title (Test set)')
plt.xlabel('Name of the X bar')
plt.ylabel('Name of the Y bar')
plt.show()
