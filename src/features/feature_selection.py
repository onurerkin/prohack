from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


def Feature_selection(ds):

    sel = RandomForestRegressor(n_estimators = 153)
    sel.fit(ds.X_train, ds.y_train)

    # Print the name and gini importance of each feature
    sel_feature=list(zip(list(ds.X_train.columns), sel.feature_importances_))
    # sfm = SelectFromModel(sel, threshold=0.15)
    # sfm.fit(ds.X_train, ds.y_train)
    # # Print the names of the most important features
    # for feature_list_index in sfm.get_support(indices=True):
    #     print(ds.X_train.columns[feature_list_index])
        
    return sel_feature




