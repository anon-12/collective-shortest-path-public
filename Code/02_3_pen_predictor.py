'''
2.3 Penalty Predictor

About Code - Train path level penalty predictor
'''

#Import Modules
import os
import pandas as pd
import ast
import time
import pickle
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

#Experiment Parameters
exp = 5
world = 'NY'
    
# Read in paramters

parameters = pd.read_csv('parameters.csv',index_col = 0).to_dict()
t_dom = parameters['Value']['t_dom']
t_max = int(((1/t_dom) * 5 -1))

#Repository Path
path = ""
os.chdir(path)
from csp_algorithms import *
from csp_toolset import *
#Path for specific experiments
exp_path = "/.../"+str(world)+"/Experiment_"+str(exp)
#Path to training data sets
train_queries_path = exp_path + "/Data/Training Data/"
#Path to output from training
train_data_path = exp_path + "/Learning/Training Data/"
#Path where high level results are captured
results_path = "/.../" + str(world) + "/"


#%% -------- Penalty Predictor --------

cpu_start_time = time.time()

#Data Input

plr_training = pd.read_csv(train_data_path + 'path_level_training_data.csv',index_col = 0)

one_hot_u = pd.get_dummies(plr_training['U Node'])
one_hot_v = pd.get_dummies(plr_training['V Node'])

X = one_hot_u.join(one_hot_v)
X['t'] = plr_training['T']

X = X.values
y = pd.DataFrame(plr_training['Penalty']).values

#Split test and training datasets
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.15, random_state=37)

# Train model

epochs = 10
batch_size = 20

penalty_model = Sequential()
penalty_model.add(Dense(4, input_dim=X_train.shape[1],  kernel_initializer='normal', activation='elu'))
penalty_model.add(Dense(50, activation='elu'))
penalty_model.add(Dense(y_train.shape[1], kernel_initializer='normal'))
# Compile model
penalty_model.compile(loss='mean_squared_error', optimizer='adam')

hist = penalty_model.fit(X_train, y_train,
          batch_size=batch_size, epochs=epochs, verbose=0)

# RMSE
# pen_rmses = np.sqrt(mean_squared_error(y_test, penalty_model.predict(X_test)))

# Correlation Coefficient
pen_corrs = np.corrcoef(y_test.T, penalty_model.predict(X_test).T)[1,0]

# R2 score
# pen_r2_scores = r2_score(y_test, penalty_model.predict(X_test))

# Output Model

penalty_model.save(exp_path+'/Learning/pen_predictor.h5')

cpu_end_time = time.time()

pen_predictor_cpu_time = cpu_end_time - cpu_start_time

#%% Apply Penalty Model to Query Set

cpu_start_time = time.time()

queries = pd.read_csv(exp_path+'/Data/queries.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

one_hot_u_queries = pd.DataFrame(columns = one_hot_u.columns)
one_hot_v_queries = pd.DataFrame(columns = one_hot_v.columns)

zero_array_u = np.zeros(one_hot_u_queries.shape[1], dtype=int)
zero_array_v = np.zeros(one_hot_v_queries.shape[1], dtype=int)

for q in range(0,queries.shape[0]):
    one_hot_u_queries.loc[q] = zero_array_u
    one_hot_v_queries.loc[q] = zero_array_v
    one_hot_u_queries.xs(q)['u_'+ str(queries.iloc[q][0])] = 1
    one_hot_v_queries.xs(q)['v_'+ str(queries.iloc[q][1])] = 1

X_queries = one_hot_u_queries.join(one_hot_v_queries)

X_queries['t'] = queries['t'].values

X_queries = X_queries.values.astype(float)

y_pred = penalty_model.predict(X_queries)

queries['Predicted Pen'] = y_pred

sub_folder_data = exp_path + '/Data'

queries.to_csv(sub_folder_data+'/queries_all.csv')

cpu_end_time = time.time()

apply_pen_predictor_cpu_time = cpu_end_time - cpu_start_time

#%%

metrics = pd.read_csv(results_path+'Offline Learning Metrics.csv',index_col = 0).astype(np.float16)
metrics_update = metrics.copy(deep=False)
metrics_update['Penalty Predictor CPU'][exp] = pen_predictor_cpu_time
metrics_update['Apply Penalty Predictor CPU'][exp] = apply_pen_predictor_cpu_time
metrics_update['Pen Predictor Corr Coeff'][exp] = pen_corrs
metrics_update.to_csv(results_path+'Offline Learning Metrics.csv')