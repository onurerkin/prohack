from scipy.optimize import linprog
import pandas as pd
import numpy as np
from cvxopt import matrix
from cvxopt import glpk

#import datasets
train=pd.read_csv('/Users/busebalci/Dev/prohack/data/raw/train.csv')

#Create exist_exp_index_binary(i) column
train['exist_exp_index_binary']=0
train['exist_exp_index_binary'][train['existence expectancy index']<0.7]=1


#Create potential for increase in the index column
train['Potential for increase in the Index']=0
train['Likely increase in the Index Coefficient']=0
for i in range(len(train)):
    train['Potential for increase in the Index'].iloc[i]=-np.log(train['existence expectancy index'].iloc[i]+0.01)+3
    train['Likely increase in the Index Coefficient'].iloc[i]=-((train['Potential for increase in the Index'].iloc[i])**2/1000)    
train['Likely increase in the Index Coefficient']=train['Likely increase in the Index Coefficient'].fillna(0)

#OPTIMIZATION

#Decision Variables
#x(i): amount of energy to allocate for each galaxy
# X matrix
var_list = list(train['galaxy'])

#Objective function
#max the total likely increase in index
# Objective function coefficients
c =  list(train['Likely increase in the Index Coefficient']) # construct a cost function

#Constraints
#1: sum(x(i))<=50000
#2: for each i x(i)<=100
#3: for each i x(i)>=5000*exist_exp_index_binary(i)

# Inequality equations with upper bound, LHS-Constraints 1 and 3
A_ineq = [np.ones(len(train)) , list(train['exist_exp_index_binary'])]

# Inequality equations with upper bound, RHS-Constraints 1 and 3
B_ineq = [50000]+[5000]

# Define bounds-Constraint #2 
bounds= (((0,100), ) * len(train)) 

#Run optimization problem
print('WITHOUT BOUNDS')
# pass these matrices to linprog, use the method 'interior-point'. '_ub' implies the upper-bound or
# inequality matrices and '_eq' imply the equality matrices
res_no_bounds = linprog(c, A_ub=A_ineq, b_ub=B_ineq ,bounds=bounds, method='interior-point')
print(res_no_bounds)

#Create "pred_opt"
pred_opt=res_no_bounds['x']