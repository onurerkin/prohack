# -*- coding: utf-8 -*-
import statistics
full_data['y_opt_pred']=np.array(y_opt_pred)
middle=full_data.iloc[int(round(len(y_opt_pred)/3, 0)):2*int(round(len(y_opt_pred)/3, 0))]
middle['y_opt_pred'][middle['existence_expectancy_index']<0.7]=statistics.mean(middle['y_opt_pred'][middle['existence_expectancy_index']<0.7])
middle['y_opt_pred'][middle['existence_expectancy_index']>=0.7]=statistics.mean(middle['y_opt_pred'][middle['existence_expectancy_index']>=0.7])

first_part =full_data.iloc[:int(round(len(y_opt_pred)/3, 0))]
last_part=full_data.iloc[2*int(round(len(y_opt_pred)/3, 0)):]

final_data=first_part.append(middle)
final_data=final_data.append(last_part)



y_opt_pred=np.array(final_data['y_opt_pred'])



