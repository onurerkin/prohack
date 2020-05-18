import pandas as pd

train = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/raw/train.csv')
train.columns = train.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')',
                                                                                                             '').str.replace(
    ',', '').str.replace('%', 'percentage').str.replace('–', '_')



train.to_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/train_column_names_fixed.csv', index=False)




test = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/raw/test.csv')
test.columns = test.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')',
                                                                                                             '').str.replace(
    ',', '').str.replace('%', 'percentage').str.replace('–', '_')

test.to_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/test_column_names_fixed.csv', index=False)

