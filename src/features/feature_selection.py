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