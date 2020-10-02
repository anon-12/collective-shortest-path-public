'''
2.2 Load Predictor

About Code - Train edge level load classifer
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

#%% Load Predictor Classifier

cpu_start_time = time.time()

# Pre-Process Data

u_nodes = []
v_nodes = []
times = []
loads = []

#%%

cpu_start_time = time.time()

# Pre-Process Data

u_nodes = []
v_nodes = []
times = []
loads = []

for elm in range(1,11):
    train_elm = pd.read_csv(str(train_data_path) + 'elm_'+str(elm)+'.csv')
    for index,row in train_elm.iterrows():
        t = 0
        for l in list(row)[2:]:
            u_nodes.append('u_' + str(row['source']))
            v_nodes.append('v_' + str(row['target']))
            times.append(t)
            loads.append(l)
            t += 1

#%%

# for elm in range(1,11):
#     train_elm = pd.read_csv(str(train_data_path) + 'elm_'+str(elm)+'.csv')
#     for i in range(0,train_elm.shape[0]):
#         for t in range(0,t_max+1):
#             # new_row = [train_elm.iloc[i][0],train_elm.iloc[i][1],t,train_elm.iloc[i][2+t]]
#             u_nodes.append('u_' + str(train_elm.iloc[i][0]))
#             v_nodes.append('v_' + str(train_elm.iloc[i][1]))
#             times.append(t)
#             loads.append(train_elm.iloc[i][2+t])

#%%
one_hot_u = pd.get_dummies(u_nodes)
one_hot_v = pd.get_dummies(v_nodes)

#%%
X = one_hot_u.join(one_hot_v)

#%%
X['t'] = times

y = pd.DataFrame(loads,columns = ['Load'])

X = X.values
y = y.values

#Split test and training datasets
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.15, random_state=37)

# Train Model

epochs = 70
batch_size = 40

model = Sequential()
model.add(Dense(4, input_dim=X_train.shape[1],  kernel_initializer='normal', activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(y_train.shape[1], kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

hist = model.fit(X_train, y_train,
          batch_size=batch_size, epochs=epochs, verbose=0)


# RMSE
# load_rmses = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

# Correlation Coefficient
load_corrs = np.corrcoef(y_test.T, model.predict(X_test).T)[1,0]

# R2 score
# load_r2_scores = r2_score(y_test, model.predict(X_test))

# Output Model

model.save(exp_path+'/Learning/load_predictor.h5')

cpu_end_time = time.time()

load_predictor_cpu_time = cpu_end_time - cpu_start_time

#%%

metrics = pd.read_csv(results_path+'/Offline Learning Metrics.csv',index_col = 0).astype(np.float16)
metrics_update = metrics.copy(deep=False)
metrics_update['Load Classifier CPU'][exp] = load_predictor_cpu_time
metrics_update['Load Predictor Corr Coeff'][exp] = load_corrs
metrics_update.to_csv(results_path+'/Offline Learning Metrics.csv')