from scipy.optimize import linprog
import pandas as pd
import numpy as np
from cvxopt import matrix
from cvxopt import glpk

def optimize_mckinsey(train):
    #Create exist_exp_index_binary(i) column
    train['exist_exp_index_binary']=0
    train['exist_exp_index_binary'][train['existence_expectancy_index']<0.7]=1
    train['exist_exp_index_binary']=-train['exist_exp_index_binary']
    
    
    #Create potential for increase in the index column
    train['Potential for increase in the Index']=0
    train['Likely increase in the Index Coefficient']=0
    for i in range(len(train)):
        train['Potential for increase in the Index'].iloc[i]=-np.log(train['y_pred'].iloc[i]+0.01)+3
        train['Likely increase in the Index Coefficient'].iloc[i]=-((train['Potential for increase in the Index'].iloc[i])**2/1000)    
        
        
    # train['Likely increase in the Index Coefficient']=train['Likely increase in the Index Coefficient'].fillna(0)
    
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
    A_ineq = [np.ones(len(train)).tolist() , list(train['exist_exp_index_binary'])]
    
    # Inequality equations with upper bound, RHS-Constraints 1 and 3
    B_ineq = [50000,-5000]
    
    # Define bounds-Constraint #2 
    bounds= list(((0,100), ) * len(train))
    
    #Run optimization problem
    # pass these matrices to linprog, use the method 'interior-point'. '_ub' implies the upper-bound or
    # inequality matrices and '_eq' imply the equality matrices
    res_no_bounds = linprog(c, A_ub=A_ineq, b_ub=B_ineq ,bounds=bounds, method='revised simplex' )
    print(res_no_bounds)
    
    
    #Create "pred_opt"
    pred_opt=res_no_bounds['x']
    
    #Calculate the
    increase = sum((pred_opt*train['Potential for increase in the Index']**2)/1000)
    print('Optimal increase: .{}'.format(increase))
    
    
    #Compare with the base case
    
    #Create a naive solution 
    
    train_for_base=train.copy()
    ss = pd.DataFrame({
        'Index':train_for_base.index,
        'pred':train_for_base['y_pred'],
        'opt_pred':0, 
        'eei':train_for_base['existence_expectancy_index'] # So we can split into low and high EEI galaxies
    })
    # Fix opt_pred
    n_low_eei = ss.loc[ss.eei < 0.7].shape[0]
    n_high_eei = ss.loc[ss.eei >= 0.7].shape[0]
    ss.loc[ss.eei < 0.7, 'opt_pred'] = 99 # 66*99 = 6534 - >10%, <100 each
    ss.loc[ss.eei >= 0.7, 'opt_pred'] = (50000-99 * len(ss.loc[ss.eei < 0.7, 'opt_pred']))/n_high_eei # The rest to high eei gs
    # Leaving 5k zillion whatsits to the admin
    ss = ss.drop('eei', axis=1)
    
    train_for_base['potential_increase'] = -np.log(train_for_base['y_pred']+0.01)+3
    increase_base = sum((ss['opt_pred']*train_for_base['potential_increase']**2)/1000)
    print('Naive increase: .{}'.format(increase_base))

    return pred_opt

#import datasets and run
test=pd.read_csv('data/processed/X_test_pred_19_may.csv')
pred_opt=optimize_mckinsey(test)

test['pred_opt']=pred_opt
test_submission=test[['y_pred', 'pred_opt']]
test_submission=test_submission.rename(columns={'y_pred':'pred'})
test_submission=test_submission.rename(columns={'pred_opt':'opt_pred'})
test_submission=test_submission.reset_index()

test_submission.to_csv('data/processed/test_submission_guardians_may_19.csv', index=False)


test['existence_expectancy_index'].isna().sum()



#control


sum(test['pred_opt'][test['existence_expectancy_index']<0.7])
sum(test['pred_opt'][test['existence_expectancy_index']>0.7])

sum(test['pred_opt'][test['existence_expectancy_index']==0.7])

sum(test['pred_opt'])









