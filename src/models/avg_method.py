# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np

sol1=pd.read_csv('/Users/busebalci/Downloads/1589989540.1326604__test_submission_guardians.csv')
sol2=pd.read_csv('/Users/busebalci/Downloads/1590118029.7963948__test_submission_guardians_may_21.csv')
sol3=pd.read_csv('/Users/busebalci/Downloads/1590264789.9791567__two_sub_combined_submission_may_23.csv')
sol4=pd.read_csv('/Users/busebalci/Downloads/1590331711.4643655__two_sub_combined_submissio_RiCgCC0.csv')
sol5=pd.read_csv('/Users/busebalci/Downloads/1590372154.2554529__y_binary_may_24.csv')
last_submission=pd.read_csv('/Users/busebalci/Downloads/1590373676.153057__pred_opt_avg_may_24.csv')


#Create noise
mu, sigma = 0, 0.5
large_noise = np.absolute(np.random.normal(mu, sigma, [890,1]))

mu, sigma = 0, 0.1
small_noise = np.random.normal(mu, sigma, [890,]) 

#resemble last submission
sol_with_last_submission=sol1
sol_with_last_submission['pred']=(sol1['pred']+sol2['pred']+sol3['pred']+sol4['pred']+sol5['pred'])/5
sol_with_last_submission['opt_pred']=last_submission['opt_pred']


#Add large noise to pred small noise to opt_pred
sol_with_last_submission['pred']  = np.array(sol_with_last_submission['pred']) + large_noise

sol_with_last_submission['opt_pred'] = np.array(sol_with_last_submission['opt_pred']) + small_noise


#Fix the logical errors


sol_with_last_submission.loc[sol_with_last_submission['pred']>1, 'pred']=1

sol_with_last_submission.loc[sol_with_last_submission['opt_pred']>100, 'opt_pred']=100 
sol_with_last_submission.loc[sol_with_last_submission['opt_pred']<0, 'opt_pred']=0

sol_with_last_submission.to_csv('/Users/busebalci/Downloads/sol_with_last_submission_may_25.csv' , index=False)

#Repeat processes for the average method
sol_with_avg_pred_opt=sol_with_last_submission
sol_with_avg_pred_opt['opt_pred']=(sol1['opt_pred']+sol2['opt_pred']+sol3['opt_pred']+sol4['opt_pred']+sol5['opt_pred'])/5

sol_with_avg_pred_opt['opt_pred'] = np.array(sol_with_avg_pred_opt['opt_pred']) + small_noise
sol_with_avg_pred_opt.loc[sol_with_avg_pred_opt['opt_pred']>100, 'opt_pred']=100 
sol_with_avg_pred_opt.loc[sol_with_avg_pred_opt['opt_pred']<0, 'opt_pred']=0


sol_with_avg_pred_opt.to_csv('/Users/busebalci/Downloads/sol_with_avg_pred_opt_may_25.csv' , index=False)




