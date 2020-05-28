import pandas as pd
import numpy as np



test_submission_guardians_may_24 = pd.read_csv('data/processed/test_submission_guardians_may_24.csv')
two_sub_combined_submission_may_23 = pd.read_csv('data/processed/two_sub_combined_submission_may_23.csv')


two_sub_combined_submission_may_23_y_pred_may_24 = two_sub_combined_submission_may_23
two_sub_combined_submission_may_23_y_pred_may_24['pred'] = test_submission_guardians_may_24['pred']

two_sub_combined_submission_may_23_y_pred_may_24.to_csv('data/processed/two_sub_combined_submission_may_23_y_pred_may_24.csv',index=False)