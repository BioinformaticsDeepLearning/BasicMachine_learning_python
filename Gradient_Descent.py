#This is the Gradient descent code for optimization of any machine learning algorithm#

#@created by Dr.Alisha parveen

import numpy as np

x =np.array([1,2,3,4,5])
y= np.array([10,12,13,8,19])

def gradient_descent(x,y):
    m_curr= b_curr= 0
    iterations=10    #(change gradually like take 10, 100, 1000, 1200, 14000 and so on)
    n= len(x)
    learning_rate = 0.05   #(change gradually from 0.0001, 0.001, 0.01, 0.1, 0.005 and so on)
    
    for i in range(iterations):
        y_predicted = m_curr*x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        d_m = -(2/n)*sum(x*(y-y_predicted))
        d_b = -(2/n)*sum(y-y_predicted)
        m_curr= m_curr-learning_rate * d_m
        b_curr= b_curr-learning_rate*d_b
        print("m {}, b {}, cost {}, iterations {}".format(m_curr, b_curr, cost, i))

gradient_descent(x,y)

#Change iterations and the learning_rate parameter to get the least value of cost/error#
