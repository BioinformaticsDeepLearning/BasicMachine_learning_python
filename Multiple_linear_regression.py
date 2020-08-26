# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')

#Creating a matrix of Independent varialbe#
X = dataset.iloc[:, :-1].values

#Creating a matrix of Dependent varialbe#
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#creating one hot encoder object with categorial feature 3 indicating the 4th column#
mt = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(mt.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
#Print the constant and the coefficient# 
print(f'constant = {reg.intercept_}')
print(f'coefficient = {reg.coef_}')

# Predicting the Test set results
y_pred = reg.predict(X_test)

#Display actual and predicted values side by side#
df= pd.DataFrame(data=y_test, columns= ['y_test'])
df['y_predict'] = y_pred
print(df)

#Model evaluation#
from sklearn import metrics
r_square = metrics.r2_score(y_test, y_pred)
print('R-Square Error:', r_square)

from sklearn.metrics import mean_poisson_deviance
MPD= mean_poisson_deviance(Y_test, Y_pred)
print('Mean poisson deviance:', MPD)

#Note: In regression model, for model evaluation we use R-squared value, RMSE; RSE; MAE. for evaluation of the classification method, we use proper confusion matrix#
