from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from src.features.label_encoder import MultiColumnLabelEncoder

from sklearn.svm import LinearSVR

def Feature_selection(dataset):

    cate_cols = ['galaxy']
    dataset.X_train = MultiColumnLabelEncoder(columns=cate_cols).transform(dataset.X_train)
    dataset.X_val = MultiColumnLabelEncoder(columns=cate_cols).transform(dataset.X_val)
    dataset.X_test = MultiColumnLabelEncoder(columns=cate_cols).transform(dataset.X_test)

    
    sel = RandomForestRegressor(n_estimators = 100).fit(dataset.X_train, dataset.y_train)
    
    sel_feature=list(zip(list(dataset.X_train.columns), sel.feature_importances_))
    model = SelectFromModel(sel, prefit=True)
    set1= dataset.X_train.columns[(model.get_support())]
    
    
    #Feature selection by using selectKbets
    model = SelectKBest(f_classif, k=20).fit(dataset.X_train, dataset.y_train)
    
    set2=dataset.X_train.columns[(model.get_support())]
    
    
    selected_columns=set(list(set1)+list(set2))


    return selected_columns

def Feature_selection_new(ds):

    # ds.X_train['y']=ds.y_train
    # ds.X_test['y']=ds.y_test
    
    galaxy_train=ds.X_train['galaxy']
    galaxy_val=ds.X_val['galaxy']
    galaxy_test=ds.X_test['galaxy']
    
    
    ds.X_train=ds.X_train.drop(columns=['galaxy'])
    ds.X_val=ds.X_val.drop(columns=['galaxy'])
    
    # from featexp import get_trend_stats
    # stats = get_trend_stats(data=ds.X_train, target_col='y', data_test=ds.X_test)
    
    import pandas as pd
    from numpy import loadtxt
    from xgboost import XGBRegressor
    from xgboost import plot_importance
    from matplotlib import pyplot
    from sklearn.metrics import mean_squared_error
    from numpy import sort
    from sklearn.feature_selection import SelectFromModel
    from math import sqrt 
    
    # fit model no training data
    model = XGBRegressor()
    model.fit(ds.X_train, ds.y_train)
    # plot feature importance
    plot_importance(model)
    pyplot.show()
    
    # make predictions for test data and evaluate
    y_pred = model.predict(ds.X_val)
    rmse = sqrt(mean_squared_error(y_pred,ds.y_val))
    print("mse: %.7f%%" % (rmse))
    # Fit model using each importance as a threshold
    thresholds = sort(model.feature_importances_)
    
    columns=list(ds.X_train.columns)
    importances=model.feature_importances_.tolist()
    column_importances=list(zip(columns,importances))
    column_importances_df = pd.DataFrame(column_importances, columns=['columns', 'importances'])

    
    result=[]
    
    for thresh in thresholds:
       	# select features using threshold
           selection = SelectFromModel(model, threshold=thresh, prefit=True)
           select_X_train = selection.transform(ds.X_train)
       	# train model
           selection_model = XGBRegressor()
           selection_model.fit(select_X_train, ds.y_train)
       	# eval model
           select_X_test = selection.transform(ds.X_val)
           y_pred = selection_model.predict(select_X_test)
           rmse = sqrt(mean_squared_error(y_pred,ds.y_val))
           result.append([thresh,select_X_train.shape[1], rmse])
        	#print("Thresh=%.7f, n=%d, mse: %.7f%%" % (thresh, select_X_train.shape[1], rmse))
           
    columns_to_drop=list(column_importances_df['columns'][column_importances_df['importances']<=0.0007036648])    
    
    ds.X_train = ds.X_train.drop(columns=columns_to_drop, axis=1)
    ds.X_train['galaxy']=galaxy_train
    ds.X_val = ds.X_val.drop(columns=columns_to_drop, axis=1) 
    ds.X_val['galaxy']=galaxy_val
    ds.X_test = ds.X_test.drop(columns=columns_to_drop, axis=1)
    
    return ds 

    
