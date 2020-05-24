import pandas as pd
import numpy as np

two_sub_combined = pd.read_csv('/Users/onurerkinsucu/Downloads/two_sub_combined.csv')
X_test = pd.read_csv('data/processed/test_column_names_fixed.csv')



cols = ['prob_0', 'prob_1', 'energy', 'pred', 'opt_pred']

two_sub_combined = two_sub_combined[cols]

two_sub_combined['combined_decision'] = 999

two_sub_combined.loc[
    (two_sub_combined['energy'] > 95) & (two_sub_combined['opt_pred'] == 100), 'combined_decision'] = 100
two_sub_combined.loc[(two_sub_combined['energy'] < 5) & (two_sub_combined['opt_pred'] == 0), 'combined_decision'] = 0

two_sub_combined.loc[two_sub_combined['combined_decision'] == 999, 'combined_decision'] = (
            two_sub_combined['energy'] + two_sub_combined['opt_pred']) / 2


coef = sum(two_sub_combined['combined_decision']) / 50000
two_sub_combined.loc[two_sub_combined['combined_decision'] != 100,'combined_decision'] = two_sub_combined['combined_decision'] * 0.5903

print(sum(two_sub_combined['combined_decision']))

two_sub_combined_submission = two_sub_combined[['pred', 'combined_decision']]
two_sub_combined_submission = two_sub_combined_submission.rename(columns={'combined_decision': 'opt_pred'})
two_sub_combined_submission = two_sub_combined_submission.reset_index()

two_sub_combined_submission.to_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/two_sub_combined_submission_may_23.csv', index=False)



X_test['opt_pred'] = two_sub_combined_submission['opt_pred']
belows = X_test[X_test['existence_expectancy_index'] < 0.7]
sum(belows['opt_pred'])
