# Importing the important libraries #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing a dataset in CSV format#
data = pd.read_csv('Data.csv')

#Creating a matrix of Independent varialbe#
X= data.iloc[:, :-1].values
print(X)

#Creating a matrix of Dependent varialbe#
Y= dataset.iloc[:, 3].values
print(Y)

#HANDLING MISSING DATA#
#Load necessary Libraries#
from sklearn.impute import SimpleImputer 

#Creating an Object called impute with their relevent parameters#
impute= SimpleImputer(missing_values = np.nan, strategy='mean')

#fit the impute object into X (Independent variable), because Independent variable contains missing values#
impute.fit(X[:, 1:3]) 
X[:, 1:3] = impute.transform(X[:, 1:3])
print(X)

#HANDLING CATEGORIAL DATA#
#Load necessary Libraries#
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#Encode Independent Variable#
#According to the output#
#france=0#
#Spain=1#
#Germany=2 which shows Germany is greater than spain according to mathematics, However it is not true, we are only categorizing for training. Therefore, to prevent this we need dummy variable#

mt = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(mt.fit_transform(X) )
print(X)

#Encode Dependent Variable#
from sklearn.preprocessing import LabelEncoder
lel = LabelEncoder()
Y = lel.fit_transform(Y)
print(Y)

# SPLITTING THE TEST AND TRAINING DATASET#
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# FEATURE SCALING#
#Different ways of feature scaling: Standardisation and Normalization#
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) #in training set, it is importance to do fit and then transform; unlike Test dataset#
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)
