# -*- coding: utf-8 -*-

import pandas as pd

full_data=pd.read_csv('/Users/busebalci/Downloads/X_train_pred_h2o_23_may.csv')
y_pred=full_data['y_pred']

####RESULTSSSS

#full_data=ds.X_val.copy()

#Get y_pred from dnn
#cate_cols = ['galaxy']
#y_pred=predict(model, X_test=full_data, cate_cols=cate_cols)
#y_pred = pd.DataFrame(y_pred,columns=['y_pred'])
y_pred=np.array(y_pred)
#Define y_true

y_true=full_data['y']
y_true=np.array(y_true)

#Get y_opt_true and y_opt_pred
full_data['y_pred']=y_true 
y_opt_true=optimize_mckinsey(full_data)
full_data['y_pred']=y_pred
y_opt_pred=optimize_mckinsey(full_data)


loss1=0.8*sqrt(mean_squared_error(y_true,y_pred))
loss2=0.2*0.01*sqrt(mean_squared_error(y_opt_true,y_opt_pred))                                

    
total_loss=loss1+loss2

y_pred = pd.DataFrame(y_pred,columns=['y_pred'])
y_true=pd.DataFrame(y_true,columns=['y_true'])
y_opt_true=pd.DataFrame(y_opt_true,columns=['y_opt_true'])
y_opt_pred=pd.DataFrame(y_opt_pred,columns=['y_opt_pred'])

frames=[ y_true,y_pred, y_opt_true, y_opt_pred]
result = pd.concat(frames, axis=1)

num_mistakes=len(result[result['y_opt_true']!=result['y_opt_pred']])

