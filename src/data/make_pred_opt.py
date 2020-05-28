import pandas as pd
import numpy as np


X_test_pred_h2o_24_may_binary = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/X_test_pred_h2o_24_may_binary.csv')
two_sub_combined_submission_may_23_y_pred_may_24 = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/two_sub_combined_submission_may_23_y_pred_may_24.csv')

new_pred_opt_df = two_sub_combined_submission_may_23_y_pred_may_24
new_pred_opt_df['p1'] = X_test_pred_h2o_24_may_binary['p1']
new_pred_opt_df['optimization_result'] = 0
new_pred_opt_df.loc[(new_pred_opt_df['pred']<0.05251), 'optimization_result'] = 100
sum(new_pred_opt_df['optimization_result'])


new_pred_opt_df['new_pred_opt'] = 999

prob_cutoff = 0.99
new_pred_opt_df.loc[(new_pred_opt_df['pred']<0.033) & (new_pred_opt_df['p1']>prob_cutoff), 'new_pred_opt'] = 100
new_pred_opt_df.loc[(new_pred_opt_df['pred']>0.065) & (new_pred_opt_df['p1']<(1-prob_cutoff)), 'new_pred_opt'] = 0
new_pred_opt_df.loc[new_pred_opt_df['new_pred_opt'] == 999, 'new_pred_opt'] = (new_pred_opt_df['optimization_result'] + new_pred_opt_df['p1']*100) / 2 *0.9057
new_pred_opt_df['pred_opt_avg'] =  (new_pred_opt_df['new_pred_opt'] + new_pred_opt_df['opt_pred'] )/ 2
print(sum(new_pred_opt_df['pred_opt_avg']))


new_pred_opt_df['optimization_result']

new_pred_opt_df = new_pred_opt_df[['index', 'pred', 'pred_opt_avg']]
new_pred_opt_df.columns = ['index', 'pred', 'opt_pred']
sum(new_pred_opt_df['opt_pred'])

new_pred_opt_df.to_csv('data/processed/pred_opt_avg_may_24.csv', index=False)


new_pred_opt_df.loc[new_pred_opt_df['p1']<0.99805, 'new_pred_opt'] = 0
print(sum(new_pred_opt_df['new_pred_opt']*100))

new_pred_opt_df = new_pred_opt_df.sort_values('pred').reset_index(drop=True)

len(new_pred_opt_df[(new_pred_opt_df['pred']<0.05251) & (new_pred_opt_df['new_pred_opt']==0)])


sum(X_test_pred_h2o_24_may_binary['predict'])


62/890



import pandas as pd
import numpy as np


X_test_pred_h2o_24_may_binary = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/X_test_pred_h2o_24_may_binary.csv')
two_sub_combined_submission_may_23_y_pred_may_24 = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/two_sub_combined_submission_may_23_y_pred_may_24.csv')

new_pred_opt_df = two_sub_combined_submission_may_23_y_pred_may_24
new_pred_opt_df['p1'] = X_test_pred_h2o_24_may_binary['p1']
coef = 50000/sum(new_pred_opt_df['p1']*100)
new_pred_opt_df['p1'] = new_pred_opt_df['p1'] * coef * 100
new_pred_opt_df = new_pred_opt_df[['index', 'pred', 'p1']]
new_pred_opt_df.columns = ['index', 'pred', 'opt_pred']
sum(new_pred_opt_df['opt_pred'])

new_pred_opt_df.to_csv('data/processed/only_classification_scaled_may_24.csv', index=False)
