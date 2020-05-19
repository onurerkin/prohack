from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVR

def Feature_selection(ds):
    
    sel = RandomForestRegressor(n_estimators = 100).fit(ds.X_train, ds.y_train)
    
    sel_feature=list(zip(list(ds.X_train.columns), sel.feature_importances_))
    model = SelectFromModel(sel, prefit=True)
    set1= ds.X_train.columns[(model.get_support())]
    
    
    #Feature selection by using selectKbets
    model = SelectKBest(f_classif, k=20).fit(ds.X_train, ds.y_train)
    
    set2=ds.X_train.columns[(model.get_support())]
    
    
    selected_columns=set(list(set1)+list(set2))


    return selected_columns