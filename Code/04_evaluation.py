# CSP Evaluation

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import math

#Repository Path
path = ""
os.chdir(path)

#Path where to land outputs
output_path = ''

# Parameter

#Add Query Set Size Field
query_set_dict = {'1':int(10000), '2':int(25000), '3':int(50000), '4':int(100000), '8':int(2000), '9':int(5000)}

#Experiment Name Labels
experiment_labels = {'Naive Dijkstra':'01. ND','Collective Top k':'02. TLATk','Load Aware Dijkstra':'05. TLAD','A-Star':'03. TLAA*','Coll A-Star Parallel':'04. CS-MAT','Coll A-Star':'04. CS-MAT'}

#Congestion Level Labels Porto
congestion_labels_porto = {'1':'03. High', '2':'04. Very High', '8':'01. Low', '9':'02. Medium'}

#Congestion Level Labels Porto
congestion_labels_ny = {'1':'01. Low', '2':'02. Medium','3':'03. High', '4':'04. Very High'}

# Display Parameter

nd_params = ['royalblue','v']
tlatk_params = ['darkorange','^']
tlaa_params = ['green',"8"]
tlaa_pp_params = ['purple',"p"]
cs_mat_params = ['tomato',"P"]

plt.rcParams.update({'font.size': 16})

#%% Average Journey Times

#Porto

world = 'Porto'

#Read in data

results = pd.read_csv('/.../'+str(world)+'/Full Results.csv')

#Drop IPR Heuristic

results = results[results['Algorithm'] != 'Impactful Path Replacement Penalty Heuristic']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.1']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.01']

#Add Query Set Size Label

# results['Query Set Size'] = int(0)
results['Query Set Size'] = results['Experiment'].map(query_set_dict)

# Derive Fields

# Actual Travel Time

results['Actual Travel Time'] = results['Total Travel Time'] * 360

# Actual Congestion Penalty

results['Actual Congestion Penalty'] = results['Total Congestion Penalty'] * 360

# Average Journey Time

results['Average Journey Time'] = (results['Actual Travel Time'] / results['Query Set Size']) / 60

# Average Congstion Penalty

results['Congestion Flow Percent'] = (results['Actual Congestion Penalty'] / results['Actual Travel Time']) * 100

#ReLabel Experiment Names

results['Algorithm Label'] = results['Algorithm'].map(experiment_labels)

#Add Label for congestion level

results['Congestion Level'] = results['Experiment'].map(congestion_labels_porto)

# Select only experiments 1,2,8,9

results = results[results['Experiment'].isin(['1','2','8','9'])]

# Remove Naive Dijkstra

results = results[~results['Algorithm'].isin(['Load Aware Dijkstra'])]

# Plot Average Time Results

fig, ax = plt.subplots(figsize=(6,3))
fig_results = results[results['Sort']  == 'Time'].pivot('Query Set Size','Algorithm Label','Average Journey Time')
fig_results.columns = ['ND','TLATk','TLAA*']
fig_results.index = fig_results.index.map(int)
fig_results.plot(kind='bar',ax = ax,color = [nd_params[0],tlatk_params[0],tlaa_params[0]])
ax.set_ylabel('AJT (Mins)')
plt.xticks(rotation=360)
fig.savefig(output_path+'Porto_Algs_AJT.pdf', bbox_inches = "tight")
plt.show()

#%% New York

world = 'NY'

#Read in data

results = pd.read_csv('/.../'+str(world)+'/Full Results.csv')

results['Experiment'] = results['Experiment'].astype(str)

#Drop IPR Heuristic

results = results[results['Algorithm'] != 'Impactful Path Replacement Penalty Heuristic']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.1']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.01']

#Add Query Set Size Label

results['Query Set Size'] = results['Experiment'].map(query_set_dict)

# Derive Fields

# Actual Travel Time

results['Actual Travel Time'] = results['Total Travel Time'] * 360

# Actual Congestion Penalty

results['Actual Congestion Penalty'] = results['Total Congestion Penalty'] * 360

# Average Journey Time

results['Average Journey Time'] = (results['Actual Travel Time'] / results['Query Set Size']) / 60

# Average Congstion Penalty

results['Congestion Flow Percent'] = (results['Actual Congestion Penalty'] / results['Actual Travel Time']) * 100

#ReLabel Experiment Names

results['Algorithm Label'] = results['Algorithm'].map(experiment_labels)

#Add Label for congestion level

results['Congestion Level'] = results['Experiment'].map(congestion_labels_ny)

# Select only experiments 1,2,3,4

results = results[results['Experiment'].isin(['1','2','3','4'])]

# Remove Naive Dijkstra

results = results[~results['Algorithm'].isin(['Load Aware Dijkstra'])]

# Plot Average Time Results

fig, ax = plt.subplots(figsize=(6,3))
fig_results = results[results['Sort']  == 'Time'].pivot('Query Set Size','Algorithm Label','Average Journey Time')
fig_results.columns = ['ND','TLATk','TLAA*']
fig_results.index = fig_results.index.map(int)
fig_results.plot(kind='bar',ax = ax,color = [nd_params[0],tlatk_params[0],tlaa_params[0]])
ax.set_ylabel('AJT (Mins)')
plt.xticks(rotation=360)
fig.savefig(output_path+'NY_Algs_AJT.pdf', bbox_inches = "tight")
plt.show()

#%% Insights From Synthetic Results

# Synthetic Results

world = 'Synthetic'

#Read in data

all_results = pd.read_csv('/.../'+str(world)+'/Full Results.csv')
experiment_params = pd.read_csv('/.../'+str(world)+'/Experiments.csv')
results = all_results.merge(experiment_params,left_on = 'Experiment',right_on = 'Experiment')

#Drop IPR Heuristic

results = results[results['Algorithm'] != 'Impactful Path Replacement Penalty Heuristic']
results = results[results['Algorithm'] != 'IPR (0.1)']
results = results[results['Algorithm'] != 'IPR (0.01)']

# Derive Fields

# Actual Travel Time

results['Actual Travel Time'] = results['Total Travel Time'] * 360

# Actual Congestion Penalty

results['Actual Congestion Penalty'] = results['Total Congestion Penalty'] * 360

# Average Journey Time

results['Average Journey Time'] = (results['Actual Travel Time'] / results['query_set_size']) / 60

# Average Congstion Penalty

results['Congestion Flow Percent'] = (results['Actual Congestion Penalty'] / results['Actual Travel Time']) * 100

#ReLabel Experiment Names

results['Algorithm Label'] = results['Algorithm'].map(experiment_labels)

#%% Effect of Clustering

ds = results
exp_mapping = {16: '1. No Clustering', 17: '2. Low Clustering', 18:'3. Med Clustering', 19:'4. High Clustering'}

sort_factor = 'Time'

base_line_1_alg = '01. ND'
base_line_2_alg = '02. TLATk'
alg1 = '03. TLAA*'

experiments = [16,17,18,19]
x_var = 'Clustering'

baseline_1 = ds[(ds['Sort'] == sort_factor) & (ds['Algorithm Label'] == base_line_1_alg) & (ds['Experiment'].isin(experiments))][['Experiment','CPU Time','Average Journey Time']]
baseline_1['Line Type'] = 'Baseline 1 : Naive Dijkstra'

baseline_2 = ds[(ds['Sort'] == sort_factor) & (ds['Algorithm Label'] == base_line_2_alg) & (ds['Experiment'].isin(experiments))][['Experiment','CPU Time','Average Journey Time']]
baseline_2['Line Type'] = 'Baseline 2 : LAD sort by time'

alg_test_1 = ds[(ds['Sort'] == sort_factor) & (ds['Algorithm Label'] == alg1) & (ds['Experiment'].isin(experiments))][['Experiment','CPU Time','Average Journey Time']]
alg_test_1['Line Type'] = 'Test 1 : LAD'

baseline_1['Clustering'] = baseline_1['Experiment'].map(exp_mapping)
baseline_1 = baseline_1.sort_values(by=['Clustering'])

baseline_2['Clustering'] = baseline_2['Experiment'].map(exp_mapping)
baseline_2 = baseline_2.sort_values(by=['Clustering'])

alg_test_1['Clustering'] = alg_test_1['Experiment'].map(exp_mapping)
alg_test_1 = alg_test_1.sort_values(by=['Clustering'])

#Combine all data
all_lines = pd.concat([baseline_1,baseline_2,alg_test_1])

#Repalce Values

all_lines['Clustering'] = all_lines['Clustering'].replace({'1. No Clustering':'None'})
all_lines['Clustering'] = all_lines['Clustering'].replace({'2. Low Clustering':'Low'})
all_lines['Clustering'] = all_lines['Clustering'].replace({'3. Med Clustering':'Medium'})
all_lines['Clustering'] = all_lines['Clustering'].replace({'4. High Clustering':'High'})

#
# Plot Baselines for Journey Time

fig, ax = plt.subplots(figsize=(6,2))

all_lines[all_lines['Line Type'] == 'Baseline 1 : Naive Dijkstra'].plot(kind = 'line', x = x_var, y = 'Average Journey Time',label = 'ND', ax = ax, marker = nd_params[1],color = nd_params[0])
all_lines[all_lines['Line Type'] == 'Baseline 2 : LAD sort by time'].plot(kind = 'line', x = x_var, y = 'Average Journey Time',label = 'TLATk', ax = ax, marker = tlatk_params[1],color = tlatk_params[0])
all_lines[all_lines['Line Type'] == 'Test 1 : LAD'].plot(kind = 'line', x = x_var, y = 'Average Journey Time',label = 'TLAA*', ax = ax, marker = tlaa_params[1], color = tlaa_params[0])
ax.get_legend().remove()
plt.ylabel('AJT (Mins)')

fig.savefig(output_path+'Synthetic_Baselines_AJT_Clustering.pdf', bbox_inches = "tight")
plt.show()

#%% Grid Size Effect

ds = results
exp_mapping = {2: '1. Small Grid', 16: '2. Medium Grid', 20:'3. Large Grid'}

sort_factor = 'Time'

base_line_1_alg = '01. ND'
base_line_2_alg = '02. TLATk'
alg1 = '03. TLAA*'

experiments = [2,16,20]
x_var = 'Grid Size'

baseline_1 = ds[(ds['Sort'] == sort_factor) & (ds['Algorithm Label'] == base_line_1_alg) & (ds['Experiment'].isin(experiments))][['Experiment','CPU Time','Average Journey Time']]
baseline_1['Line Type'] = 'Baseline 1 : Naive Dijkstra'

baseline_2 = ds[(ds['Sort'] == sort_factor) & (ds['Algorithm Label'] == base_line_2_alg) & (ds['Experiment'].isin(experiments))][['Experiment','CPU Time','Average Journey Time']]
baseline_2['Line Type'] = 'Baseline 2 : LAD sort by time'

alg_test_1 = ds[(ds['Sort'] == sort_factor) & (ds['Algorithm Label'] == alg1) & (ds['Experiment'].isin(experiments))][['Experiment','CPU Time','Average Journey Time']]
alg_test_1['Line Type'] = 'Test 1 : LAD'

baseline_1[x_var] = baseline_1['Experiment'].map(exp_mapping)
baseline_1 = baseline_1.sort_values(by=[x_var])

baseline_2[x_var] = baseline_2['Experiment'].map(exp_mapping)
baseline_2 = baseline_2.sort_values(by=[x_var])

alg_test_1[x_var] = alg_test_1['Experiment'].map(exp_mapping)
alg_test_1 = alg_test_1.sort_values(by=[x_var])


# Combine

all_lines = pd.concat([baseline_1,baseline_2,alg_test_1])

all_lines[x_var] = all_lines[x_var].replace({'1. Small Grid':'Small'})
all_lines[x_var] = all_lines[x_var].replace({'2. Medium Grid':'Medium'})
all_lines[x_var] = all_lines[x_var].replace({'3. Large Grid':'Large'})

# Plot Baselines for Journey Time

fig, ax = plt.subplots(figsize=(6,2))

all_lines[all_lines['Line Type'] == 'Baseline 1 : Naive Dijkstra'].plot(kind = 'line', x = x_var, y = 'Average Journey Time',label = 'ND', ax = ax, marker = nd_params[1], color = nd_params[0])
all_lines[all_lines['Line Type'] == 'Baseline 2 : LAD sort by time'].plot(kind = 'line', x = x_var, y = 'Average Journey Time',label = 'TLATk', ax = ax, marker = tlatk_params[1], color = tlatk_params[0])
all_lines[all_lines['Line Type'] == 'Test 1 : LAD'].plot(kind = 'line', x = x_var, y = 'Average Journey Time',label = 'TLAA*', ax = ax, marker = tlaa_params[1], color = tlaa_params[0])
plt.ylabel('AJT (Mins)')

fig.savefig(output_path+'Synthetic_Baselines_AJT_Grid_Size.pdf', bbox_inches = "tight")
plt.show()

#%% Basline Comparisons - Query Length (3,2,4)

ds = results
exp_mapping = {3: '1. Small', 2: '2. Average', 4:'3. Large'}

sort_factor = 'Time'

base_line_1_alg = '01. ND'
base_line_2_alg = '02. TLATk'
alg1 = '03. TLAA*'

experiments = [3,2,4]
x_var = 'Query Length'

baseline_1 = ds[(ds['Sort'] == sort_factor) & (ds['Algorithm Label'] == base_line_1_alg) & (ds['Experiment'].isin(experiments))][['Experiment','CPU Time','Average Journey Time']]
baseline_1['Line Type'] = 'Baseline 1 : Naive Dijkstra'

baseline_2 = ds[(ds['Sort'] == sort_factor) & (ds['Algorithm Label'] == base_line_2_alg) & (ds['Experiment'].isin(experiments))][['Experiment','CPU Time','Average Journey Time']]
baseline_2['Line Type'] = 'Baseline 2 : LAD sort by time'

alg_test_1 = ds[(ds['Sort'] == sort_factor) & (ds['Algorithm Label'] == alg1) & (ds['Experiment'].isin(experiments))][['Experiment','CPU Time','Average Journey Time']]
alg_test_1['Line Type'] = 'Test 1 : LAD'

baseline_1[x_var] = baseline_1['Experiment'].map(exp_mapping)
baseline_1 = baseline_1.sort_values(by=[x_var])

baseline_2[x_var] = baseline_2['Experiment'].map(exp_mapping)
baseline_2 = baseline_2.sort_values(by=[x_var])

alg_test_1[x_var] = alg_test_1['Experiment'].map(exp_mapping)
alg_test_1 = alg_test_1.sort_values(by=[x_var])


# Combine

all_lines = pd.concat([baseline_1,baseline_2,alg_test_1])

all_lines[x_var] = all_lines[x_var].replace({'1. Small':'Small'})
all_lines[x_var] = all_lines[x_var].replace({'2. Average':'Average'})
all_lines[x_var] = all_lines[x_var].replace({'3. Large':'Large'})

# Plot Baselines for Journey Time

fig, ax = plt.subplots(figsize=(6,2))

all_lines[all_lines['Line Type'] == 'Baseline 1 : Naive Dijkstra'].plot(kind = 'line', x = x_var, y = 'Average Journey Time',label = 'ND', ax = ax, marker = nd_params[1], color = nd_params[0])
all_lines[all_lines['Line Type'] == 'Baseline 2 : LAD sort by time'].plot(kind = 'line', x = x_var, y = 'Average Journey Time',label = 'TLATk', ax = ax, marker = tlatk_params[1], color = tlatk_params[0])
all_lines[all_lines['Line Type'] == 'Test 1 : LAD'].plot(kind = 'line', x = x_var, y = 'Average Journey Time',label = 'TLAA*', ax = ax, marker = tlaa_params[1], color = tlaa_params[0])
plt.ylabel('AJT (Mins)')
ax.get_legend().remove()
fig.savefig(output_path+'Synthetic_Baselines_AJT_Query_Length.pdf', bbox_inches = "tight")
plt.show()

#%% Testing Heuristic by Average Journey Time

#%% Porto
world = 'Porto'

#Read in data

results = pd.read_csv('/.../'+str(world)+'/Full Results.csv')

#Drop IPR Heuristic

results = results[results['Algorithm'] != 'Impactful Path Replacement Penalty Heuristic']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.1']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.01']

#Add Query Set Size Label

results['Query Set Size'] = results['Experiment'].map(query_set_dict)

# Derive Fields

# Actual Travel Time

results['Actual Travel Time'] = results['Total Travel Time'] * 360

# Actual Congestion Penalty

results['Actual Congestion Penalty'] = results['Total Congestion Penalty'] * 360

# Average Journey Time

results['Average Journey Time'] = (results['Actual Travel Time'] / results['Query Set Size']) / 60

# Average Congstion Penalty

results['Congestion Flow Percent'] = (results['Actual Congestion Penalty'] / results['Actual Travel Time']) * 100

#ReLabel Experiment Names

results['Algorithm Label'] = results['Algorithm'].map(experiment_labels)

#Add Label for congestion level

results['Congestion Level'] = results['Experiment'].map(congestion_labels_porto)

# Select only experiments 1,2,8,9

results = results[results['Experiment'].isin(['1','2','8','9'])]

# Get Data for Plot

nd = results[(results['Algorithm Label'] == '01. ND') & (results['Sort'] == 'Time')].pivot('Query Set Size','Algorithm Label','Average Journey Time')
nd.columns = ['ND']
nd.index = nd.index.map(int)

TLATk = results[(results['Algorithm Label'] == '02. TLATk') & (results['Sort'] == 'Time')].pivot('Query Set Size','Algorithm Label','Average Journey Time')
TLATk.columns = ['TLATk']
TLATk.index = TLATk.index.map(int)

laa_time = results[(results['Algorithm Label'] == '03. TLAA*') & (results['Sort'] == 'Time')].pivot('Query Set Size','Algorithm Label','Average Journey Time')
laa_time.columns = ['TLAA*']
laa_time.index = laa_time.index.map(int)

laa_time_pred = results[(results['Algorithm Label'] == '03. TLAA*') & (results['Sort'] == 'Time then Pen Desc')].pivot('Query Set Size','Algorithm Label','Average Journey Time')
laa_time_pred.columns = ['TLAA*-PP']
laa_time_pred.index = laa_time_pred.index.map(int)

coll_la = results[(results['Algorithm Label'] == '04. CS-MAT') & (results['Sort'] == 'No Sort')].pivot('Query Set Size','Algorithm Label','Average Journey Time')
coll_la.columns = ['CS-MAT']
coll_la.index = coll_la.index.map(int)

# Plot

fig, ax = plt.subplots(figsize=(8,3))

nd.plot(ax = ax, marker = nd_params[1], color = nd_params[0])
TLATk.plot(ax = ax, marker = tlatk_params[1], color = tlatk_params[0])
laa_time.plot(ax = ax, marker = tlaa_params[1], color = tlaa_params[0])
laa_time_pred.plot(ax = ax, marker = tlaa_pp_params[1], color = tlaa_pp_params[0])
coll_la.plot(ax = ax, marker = cs_mat_params[1], color = cs_mat_params[0])
ax.set_ylabel('AJT (Minutes)')
ax.get_legend().remove()
fig.savefig(output_path+'Porto_heuristic_AJT.pdf', bbox_inches = "tight")
plt.show()

#%% New York
world = 'NY'

#Read in data

results = pd.read_csv('/.../'+str(world)+'/Full Results.csv')

#Drop IPR Heuristic

results = results[results['Algorithm'] != 'Impactful Path Replacement Penalty Heuristic']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.1']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.01']

#Add Query Set Size Label

results['Query Set Size'] = results['Experiment'].map(query_set_dict)

# Derive Fields

# Actual Travel Time

results['Actual Travel Time'] = results['Total Travel Time'] * 360

# Actual Congestion Penalty

results['Actual Congestion Penalty'] = results['Total Congestion Penalty'] * 360

# Average Journey Time

results['Average Journey Time'] = (results['Actual Travel Time'] / results['Query Set Size']) / 60

# Average Congstion Penalty

results['Congestion Flow Percent'] = (results['Actual Congestion Penalty'] / results['Actual Travel Time']) * 100

#ReLabel Experiment Names

results['Algorithm Label'] = results['Algorithm'].map(experiment_labels)

#Add Label for congestion level

results['Congestion Level'] = results['Experiment'].map(congestion_labels_ny)

# Select only experiments 1,2,8,9

results = results[results['Experiment'].isin(['1','2','3','4'])]

# Get Data for Plot

nd = results[(results['Algorithm Label'] == '01. ND') & (results['Sort'] == 'Time')].pivot('Query Set Size','Algorithm Label','Average Journey Time')
nd.columns = ['ND']
nd.index = nd.index.map(int)

TLATk = results[(results['Algorithm Label'] == '02. TLATk') & (results['Sort'] == 'Time')].pivot('Query Set Size','Algorithm Label','Average Journey Time')
TLATk.columns = ['TLATk']
TLATk.index = TLATk.index.map(int)

laa_time = results[(results['Algorithm Label'] == '03. TLAA*') & (results['Sort'] == 'Time')].pivot('Query Set Size','Algorithm Label','Average Journey Time')
laa_time.columns = ['TLAA*']
laa_time.index = laa_time.index.map(int)

laa_time_pred = results[(results['Algorithm Label'] == '03. TLAA*') & (results['Sort'] == 'Time then Pen Desc')].pivot('Query Set Size','Algorithm Label','Average Journey Time')
laa_time_pred.columns = ['TLAA*-PP']
laa_time_pred.index = laa_time_pred.index.map(int)

coll_la = results[(results['Algorithm Label'] == '04. CS-MAT') & (results['Sort'] == 'No Sort')].pivot('Query Set Size','Algorithm Label','Average Journey Time')
coll_la.columns = ['CS-MAT']
coll_la.index = coll_la.index.map(int)

# Plot

fig, ax = plt.subplots(figsize=(8,3))

nd.plot(ax = ax, marker = nd_params[1], color = nd_params[0])
TLATk.plot(ax = ax, marker = tlatk_params[1], color = tlatk_params[0])
laa_time.plot(ax = ax, marker = tlaa_params[1], color = tlaa_params[0])
laa_time_pred.plot(ax = ax, marker = tlaa_pp_params[1], color = tlaa_pp_params[0])
coll_la.plot(ax = ax, marker = cs_mat_params[1], color = cs_mat_params[0])
ax.set_ylabel('AJT (Minutes)')

ax.legend(labelspacing=.1, prop={'size': 14})

fig.savefig(output_path+'NY_heuristic_AJT.pdf', bbox_inches = "tight")
plt.show()

#%% Utilisation Analysis

#Porto

# For ND (time), topk (time), LAA* (time then pen), collective arrival time
# Get System Level Results Over Time
# Average Utilisation for Different Congestion Levels


experiments = ['1','2','8','9']
algs = [['Naive Dijkstra','Time'],['Collective Top k','Time'],['A-Star','Time then Pen Desc'],['Coll A-Star','1']]


results = []

for exp in experiments:
    for alg in algs:

        system_results = pd.read_csv('/.../Porto/Experiment_'+str(exp)+'/Results/'+str(alg[0])+'/'+str(alg[1])+'/System Level Results Over Time - Exp '+str(exp)+'.csv',index_col = 0).T[:40]
        
        result = [exp,alg[0],system_results['Utilisation Residual'].mean(),system_results['Load Distribution'].mean()]
        
        results.append(result)

utilisation_results_porto = pd.DataFrame(results,columns = ['Experiment','Algorithm','Capacity Utilisation','Load Distribution'])

#ReLabel Experiment Names

utilisation_results_porto['Algorithm Label'] = utilisation_results_porto['Algorithm'].map(experiment_labels)

#Add Label for congestion level

utilisation_results_porto['Congestion Level'] = utilisation_results_porto['Experiment'].map(congestion_labels_porto)

#Get Query Set Size
utilisation_results_porto['Query Set Size'] = utilisation_results_porto['Experiment'].map(query_set_dict)

#%% New York

experiments = ['1','2','3','4']
algs = [['Naive Dijkstra','Time'],['Collective Top k','Time'],['A-Star','Time then Pen Desc'],['Coll A-Star','1']]


results = []

for exp in experiments:
    for alg in algs:

        system_results = pd.read_csv('/.../NY/Experiment_'+str(exp)+'/Results/'+str(alg[0])+'/'+str(alg[1])+'/System Level Results Over Time - Exp '+str(exp)+'.csv',index_col = 0).T[:40]
        
        result = [exp,alg[0],system_results['Utilisation Residual'].mean(),system_results['Load Distribution'].mean()]
        
        results.append(result)

utilisation_results_ny = pd.DataFrame(results,columns = ['Experiment','Algorithm','Capacity Utilisation','Load Distribution'])

#ReLabel Experiment Names

utilisation_results_ny['Algorithm Label'] = utilisation_results_ny['Algorithm'].map(experiment_labels)

#Add Label for congestion level

utilisation_results_ny['Congestion Level'] = utilisation_results_ny['Experiment'].map(congestion_labels_ny)

#Get Query Set Size

utilisation_results_ny['Query Set Size'] = utilisation_results_ny['Experiment'].map(query_set_dict)

#%%

#Plot FFCU

fig, ax = plt.subplots(1,2,figsize=(10,3))

fig_results = utilisation_results_porto.pivot('Query Set Size','Algorithm Label','Capacity Utilisation')
fig_results.columns = ['ND','TLATk','TLAA*','CS-MAT']
fig_results.index = fig_results.index.map(int)
fig_results = fig_results*100

fig_results['ND'].plot(ax = ax[0], marker = nd_params[1], color = nd_params[0])
fig_results['TLATk'].plot(ax = ax[0], marker = tlatk_params[1], color = tlatk_params[0])
fig_results['TLAA*'].plot(ax = ax[0], marker = tlaa_params[1], color = tlaa_params[0])
fig_results['CS-MAT'].plot(ax = ax[0], marker = cs_mat_params[1], color = cs_mat_params[0])

ax[0].set_ylabel('FFCU (%)')
ax[0].set_title('Porto')



fig_results = utilisation_results_ny.pivot('Query Set Size','Algorithm Label','Capacity Utilisation')
fig_results.columns = ['ND','TLATk','TLAA*','CS-MAT']
fig_results.index = fig_results.index.map(int)
ny_results = fig_results
fig_results = fig_results*100



fig_results['ND'].plot(ax = ax[1], marker = nd_params[1], color = nd_params[0])
fig_results['TLATk'].plot(ax = ax[1], marker = tlatk_params[1], color = tlatk_params[0])
fig_results['TLAA*'].plot(ax = ax[1], marker = tlaa_params[1], color = tlaa_params[0])
fig_results['CS-MAT'].plot(ax = ax[1], marker = cs_mat_params[1], color = cs_mat_params[0])

ax[1].xaxis.set_major_locator(plt.MaxNLocator(3))

# ax[1].legend()
ax[1].set_title('New York')


fig.savefig(output_path+'Utilisation_ffcu.pdf', bbox_inches = "tight")
plt.show()

#Plot LD

fig, ax = plt.subplots(1,2,figsize=(10,3))

fig_results = utilisation_results_porto.pivot('Query Set Size','Algorithm Label','Load Distribution')
fig_results.columns = ['ND','TLATk','TLAA*','CS-MAT']
fig_results.index = fig_results.index.map(int)
fig_results = fig_results*100

fig_results['ND'].plot(ax = ax[0], marker = nd_params[1], color = nd_params[0])
fig_results['TLATk'].plot(ax = ax[0], marker = tlatk_params[1], color = tlatk_params[0])
fig_results['TLAA*'].plot(ax = ax[0], marker = tlaa_params[1], color = tlaa_params[0])
fig_results['CS-MAT'].plot(ax = ax[0], marker = cs_mat_params[1], color = cs_mat_params[0])

ax[0].set_ylabel('Load Distribution %')
ax[0].set_title('Porto')


fig_results = utilisation_results_ny.pivot('Query Set Size','Algorithm Label','Load Distribution')
fig_results.columns = ['ND','TLATk','TLAA*','CS-MAT']
fig_results.index = fig_results.index.map(int)
fig_results = fig_results*100

fig_results['ND'].plot(ax = ax[1], marker = nd_params[1], color = nd_params[0])
fig_results['TLATk'].plot(ax = ax[1], marker = tlatk_params[1], color = tlatk_params[0])
fig_results['TLAA*'].plot(ax = ax[1], marker = tlaa_params[1], color = tlaa_params[0])
fig_results['CS-MAT'].plot(ax = ax[1], marker = cs_mat_params[1], color = cs_mat_params[0])

ax[1].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[1].legend(labelspacing=.2)
ax[1].set_title('New York')

fig.savefig(output_path+'Utilisation_ld.pdf', bbox_inches = "tight")
plt.show()


#%% Fairness
    
# Porto

porto_experiments = ['8','9','1','2']
algs = [['Naive Dijkstra','Time'],['Collective Top k','Time'],['A-Star','Time then Pen Desc'],['Coll A-Star','1']]

fig, ax = plt.subplots(4,4,figsize=(26,12))

across = 0

for exp in porto_experiments:
    down = 0
    print('Experiment : ' + str(exp))
    for alg in algs:

        if alg[0] == 'Naive Dijkstra':
            hist_colour = nd_params[0]
        elif alg[0] == 'Collective Top k':
            hist_colour = tlatk_params[0]
        elif alg[0] == 'A-Star':
            hist_colour = tlaa_params[0]
        elif alg[0] == 'Coll A-Star':
            hist_colour = cs_mat_params[0]
        
        plr = pd.read_csv('/.../CSP/Porto/Experiment_'+str(exp)+'/Results/'+str(alg[0])+'/'+str(alg[1])+'/Path Level Results - Exp '+str(exp)+'.csv')
        
        plr['Actual Penalty'] = round((plr['penalty']*360)/60)
        
        print('Algorithm : ' + str(alg[0]))
        print('Mean : ' + str(plr['Actual Penalty'].mean()))
        print('Std Dev : ' + str(plr['Actual Penalty'].std()))
        print('Variance : ' + str(plr['Actual Penalty'].var()))
        print('Skew : ' + str(plr['Actual Penalty'].skew()))
        print()
        
        # ax[down,across].set_title(str(experiment_labels[alg[0]][4:]) + ' - ' + str(query_set_dict[exp]) + ' Queries')

        ax[down,across].text(0.3, 0.9, str(experiment_labels[alg[0]][4:]) + ' - ' + str(query_set_dict[exp]) + ' Queries', transform=ax[down,across].transAxes)
        
        ax[down,across].hist(plr['Actual Penalty'], density=False, color = hist_colour)

        if down == 3:
            ax[down,across].set_xlabel('Congestion Penalty')
        if across == 0:
            ax[down,across].set_ylabel('User Count')
            
        if exp == '8':
            ax[down,across].set_xlim([0,12])
            ax[down,across].set_ylim([0,1750])
        elif exp == '9':
            ax[down,across].set_xlim([0,30])
            ax[down,across].set_ylim([0,2750])
        elif exp == '1':
            ax[down,across].set_xlim([0,45])
            ax[down,across].set_ylim([0,3500])
        elif exp == '2':
            ax[down,across].set_xlim([0,90])
            ax[down,across].set_ylim([0,6750])
        down += 1
    across += 1

fig.savefig(output_path+'Porto_Fairness_Histograms.pdf', bbox_inches = "tight")
plt.show()


#%% Porto

porto_experiments = ['9','1','2']
algs = [['Naive Dijkstra','Time'],['Collective Top k','Time'],['A-Star','Time then Pen Desc'],['Coll A-Star','1']]

fig, ax = plt.subplots(4,3,figsize=(26,8))

across = 0

for exp in porto_experiments:
    down = 0
    print('Experiment : ' + str(exp))
    for alg in algs:

        if alg[0] == 'Naive Dijkstra':
            hist_colour = nd_params[0]
        elif alg[0] == 'Collective Top k':
            hist_colour = tlatk_params[0]
        elif alg[0] == 'A-Star':
            hist_colour = tlaa_params[0]
        elif alg[0] == 'Coll A-Star':
            hist_colour = cs_mat_params[0]
        
        plr = pd.read_csv('/.../Porto/Experiment_'+str(exp)+'/Results/'+str(alg[0])+'/'+str(alg[1])+'/Path Level Results - Exp '+str(exp)+'.csv')
        
        plr['Actual Penalty'] = round((plr['penalty']*360)/60)
        
        print('Algorithm : ' + str(alg[0]))
        print('Mean : ' + str(plr['Actual Penalty'].mean()))
        print('Std Dev : ' + str(plr['Actual Penalty'].std()))
        print('Variance : ' + str(plr['Actual Penalty'].var()))
        print('Skew : ' + str(plr['Actual Penalty'].skew()))
        print()
        
        # ax[down,across].set_title(str(experiment_labels[alg[0]][4:]) + ' - ' + str(query_set_dict[exp]) + ' Queries')

        ax[down,across].text(0.3, 0.8, str(experiment_labels[alg[0]][4:]) + ' - ' + str(query_set_dict[exp]) + ' Queries', transform=ax[down,across].transAxes)
        
        ax[down,across].hist(plr['Actual Penalty'], density=True, color = hist_colour)

        if down == 3:
            ax[down,across].set_xlabel('Congestion Penalty')
        if across == 0:
            ax[down,across].set_ylabel('User Count')
            
        if exp == '8':
            ax[down,across].set_xlim([0,12])
            # ax[down,across].set_ylim([0,1750])
        elif exp == '9':
            ax[down,across].set_xlim([0,30])
            # ax[down,across].set_ylim([0,2750])
        elif exp == '1':
            ax[down,across].set_xlim([0,45])
            # ax[down,across].set_ylim([0,3500])
        elif exp == '2':
            ax[down,across].set_xlim([0,90])
            # ax[down,across].set_ylim([0,6750])
        down += 1
    across += 1

fig.savefig(output_path+'Porto_Fairness_Histograms.pdf', bbox_inches = "tight")
plt.show()

#%% NY - Med Congestion

algs = [['Naive Dijkstra','Time'],['Collective Top k','Time'],['A-Star','Time then Pen Desc'],['Coll A-Star','1']]


fig_size_param = (6,5)

fig, ax = plt.subplots(4,1,figsize=fig_size_param)

exp = "2"

print('Experiment : ' + str(exp))

plot = 0

for alg in algs:
    
    if alg[0] == 'Naive Dijkstra':
        hist_colour = nd_params[0]
    elif alg[0] == 'Collective Top k':
        hist_colour = tlatk_params[0]
    elif alg[0] == 'A-Star':
        hist_colour = tlaa_params[0]
    elif alg[0] == 'Coll A-Star':
        hist_colour = cs_mat_params[0]
    
    plr = pd.read_csv('/.../NY/Experiment_'+str(exp)+'/Results/'+str(alg[0])+'/'+str(alg[1])+'/Path Level Results - Exp '+str(exp)+'.csv')
    
    plr['Actual Penalty'] = round((plr['penalty']*360)/60)
    
    print('Algorithm : ' + str(alg[0]))
    print('Mean : ' + str(plr['Actual Penalty'].mean()))
    print('Std Dev : ' + str(plr['Actual Penalty'].std()))
    print('Variance : ' + str(plr['Actual Penalty'].var()))
    print('Skew : ' + str(plr['Actual Penalty'].skew()))
    print()        
    
    ax[plot].text(0.5, 0.75, str(experiment_labels[alg[0]][4:]), transform=ax[plot].transAxes)
    # ax[plot].hist(plr['Actual Penalty'], density=True, color = hist_colour)
    ax[plot].hist(plr['Actual Penalty'], color = hist_colour, weights=np.ones(len(plr['Actual Penalty'])) / len(plr['Actual Penalty']))
    
    if plot == 3:
        ax[plot].set_xlabel('Congestion Penalty')
    # ax[plot].set_ylabel('User Count')
    
    if plot != 3:
        ax[plot].set_xticks([])
        ax[plot].set_xticks([], minor=True)
    
    if exp == '1':
        ax[plot].set_xlim([0,16])
        # ax[plot].set_ylim([0,10000])
    elif exp == '2':
        ax[plot].set_xlim([0,35])
        # ax[plot].set_ylim([0,13500])
    elif exp == '3':
        ax[plot].set_xlim([0,70])
        # ax[plot].set_ylim([0,20250])
    elif exp == '4':
        ax[plot].set_xlim([0,120])
        # ax[plot].set_ylim([0,35000])
        
    vals = ax[plot].get_yticks()
    ax[plot].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plot += 1

fig.savefig(output_path+'NY_Fairness_Med_Cong.pdf', bbox_inches = "tight")
plt.show()

#%% NY - High Congestion

algs = [['Naive Dijkstra','Time'],['Collective Top k','Time'],['A-Star','Time then Pen Desc'],['Coll A-Star','1']]

fig, ax = plt.subplots(4,1,figsize=fig_size_param)

exp = "3"

print('Experiment : ' + str(exp))

plot = 0

for alg in algs:
    
    if alg[0] == 'Naive Dijkstra':
        hist_colour = nd_params[0]
    elif alg[0] == 'Collective Top k':
        hist_colour = tlatk_params[0]
    elif alg[0] == 'A-Star':
        hist_colour = tlaa_params[0]
    elif alg[0] == 'Coll A-Star':
        hist_colour = cs_mat_params[0]
    
    plr = pd.read_csv('/.../NY/Experiment_'+str(exp)+'/Results/'+str(alg[0])+'/'+str(alg[1])+'/Path Level Results - Exp '+str(exp)+'.csv')
    
    plr['Actual Penalty'] = round((plr['penalty']*360)/60)
    
    print('Algorithm : ' + str(alg[0]))
    print('Mean : ' + str(plr['Actual Penalty'].mean()))
    print('Std Dev : ' + str(plr['Actual Penalty'].std()))
    print('Variance : ' + str(plr['Actual Penalty'].var()))
    print('Skew : ' + str(plr['Actual Penalty'].skew()))
    print()        
    
    ax[plot].text(0.5, 0.75, str(experiment_labels[alg[0]][4:]), transform=ax[plot].transAxes)
    ax[plot].hist(plr['Actual Penalty'], color = hist_colour, weights=np.ones(len(plr['Actual Penalty'])) / len(plr['Actual Penalty']))
    
    if plot == 3:
        ax[plot].set_xlabel('Congestion Penalty')
    # ax[plot].set_ylabel('User Count')
    
    if plot != 3:
        ax[plot].set_xticks([])
        ax[plot].set_xticks([], minor=True)
    
    if exp == '1':
        ax[plot].set_xlim([0,16])
        # ax[down,across].set_ylim([0,10000])
    elif exp == '2':
        ax[plot].set_xlim([0,35])
        # ax[down,across].set_ylim([0,13500])
    elif exp == '3':
        ax[plot].set_xlim([0,70])
        # ax[down,across].set_ylim([0,20250])
    elif exp == '4':
        ax[plot].set_xlim([0,120])
        # ax[down,across].set_ylim([0,35000])
        
    vals = ax[plot].get_yticks()
    ax[plot].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plot += 1

fig.savefig(output_path+'NY_Fairness_High_Cong.pdf', bbox_inches = "tight")
plt.show()

#%% NY - Very High Congestion

algs = [['Naive Dijkstra','Time'],['Collective Top k','Time'],['A-Star','Time then Pen Desc'],['Coll A-Star','1']]

fig, ax = plt.subplots(4,1,figsize=fig_size_param)

exp = "4"

print('Experiment : ' + str(exp))

plot = 0

for alg in algs:
    
    if alg[0] == 'Naive Dijkstra':
        hist_colour = nd_params[0]
    elif alg[0] == 'Collective Top k':
        hist_colour = tlatk_params[0]
    elif alg[0] == 'A-Star':
        hist_colour = tlaa_params[0]
    elif alg[0] == 'Coll A-Star':
        hist_colour = cs_mat_params[0]
    
    plr = pd.read_csv('/.../NY/Experiment_'+str(exp)+'/Results/'+str(alg[0])+'/'+str(alg[1])+'/Path Level Results - Exp '+str(exp)+'.csv')
    
    plr['Actual Penalty'] = round((plr['penalty']*360)/60)
    
    print('Algorithm : ' + str(alg[0]))
    print('Mean : ' + str(plr['Actual Penalty'].mean()))
    print('Std Dev : ' + str(plr['Actual Penalty'].std()))
    print('Variance : ' + str(plr['Actual Penalty'].var()))
    print('Skew : ' + str(plr['Actual Penalty'].skew()))
    print()        
    
    ax[plot].text(0.5, 0.75, str(experiment_labels[alg[0]][4:]), transform=ax[plot].transAxes)
    ax[plot].hist(plr['Actual Penalty'], color = hist_colour, weights=np.ones(len(plr['Actual Penalty'])) / len(plr['Actual Penalty']))
    
    if plot == 3:
        ax[plot].set_xlabel('Congestion Penalty')
    # ax[plot].set_ylabel('User Count')
    
    if plot != 3:
        ax[plot].set_xticks([])
        ax[plot].set_xticks([], minor=True)
    
    if exp == '1':
        ax[plot].set_xlim([0,16])
        # ax[down,across].set_ylim([0,10000])
    elif exp == '2':
        ax[plot].set_xlim([0,35])
        # ax[down,across].set_ylim([0,13500])
    elif exp == '3':
        ax[plot].set_xlim([0,70])
        # ax[down,across].set_ylim([0,20250])
    elif exp == '4':
        ax[plot].set_xlim([0,120])
        # ax[down,across].set_ylim([0,35000])
        
    vals = ax[plot].get_yticks()
    ax[plot].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plot += 1

fig.savefig(output_path+'NY_Fairness_V_High_Cong.pdf', bbox_inches = "tight")
plt.show()

#%% New York

ny_experiments = ['1','2','3','4']
algs = [['Naive Dijkstra','Time'],['Collective Top k','Time'],['A-Star','Time then Pen Desc'],['Coll A-Star','1']]

fig, ax = plt.subplots(4,4,figsize=(30,12))

across = 0

for exp in ny_experiments:
    down = 0
    print('Experiment : ' + str(exp))
    for alg in algs:
        
        if alg[0] == 'Naive Dijkstra':
            hist_colour = nd_params[0]
        elif alg[0] == 'Collective Top k':
            hist_colour = tlatk_params[0]
        elif alg[0] == 'A-Star':
            hist_colour = tlaa_params[0]
        elif alg[0] == 'Coll A-Star':
            hist_colour = cs_mat_params[0]
        
        plr = pd.read_csv('/.../NY/Experiment_'+str(exp)+'/Results/'+str(alg[0])+'/'+str(alg[1])+'/Path Level Results - Exp '+str(exp)+'.csv')
        
        plr['Actual Penalty'] = round((plr['penalty']*360)/60)
        
        print('Algorithm : ' + str(alg[0]))
        print('Mean : ' + str(plr['Actual Penalty'].mean()))
        print('Std Dev : ' + str(plr['Actual Penalty'].std()))
        print('Variance : ' + str(plr['Actual Penalty'].var()))
        print('Skew : ' + str(plr['Actual Penalty'].skew()))
        print()        
        
        ax[down,across].text(0.3, 0.9, str(experiment_labels[alg[0]][4:]) + ' - ' + str(query_set_dict[exp]) + ' Queries', transform=ax[down,across].transAxes)
        ax[down,across].hist(plr['Actual Penalty'], density=False, color = hist_colour)
        
        if down == 3:
            ax[down,across].set_xlabel('Congestion Penalty')
        if across == 0:
            ax[down,across].set_ylabel('User Count')
            
        if exp == '1':
            ax[down,across].set_xlim([0,16])
            ax[down,across].set_ylim([0,10000])
        elif exp == '2':
            ax[down,across].set_xlim([0,35])
            ax[down,across].set_ylim([0,13500])
        elif exp == '3':
            ax[down,across].set_xlim([0,70])
            ax[down,across].set_ylim([0,20250])
        elif exp == '4':
            ax[down,across].set_xlim([0,120])
            ax[down,across].set_ylim([0,35000])
        down += 1
    across += 1

fig.savefig(output_path+'NY_Fairness_Histograms.pdf', bbox_inches = "tight")
plt.show()

#%% Fairness analysios 2

# Porto
porto_experiments = ['8','9','1','2']
algs = [['Naive Dijkstra','Time'],['Collective Top k','Time'],['A-Star','Time then Pen Desc'],['Coll A-Star','1']]

for exp in porto_experiments:
    for alg in algs:

        plr = pd.read_csv('/.../Porto/Experiment_'+str(exp)+'/Results/'+str(alg[0])+'/'+str(alg[1])+'/Path Level Results - Exp '+str(exp)+'.csv')
        plr['Actual Penalty'] = round((plr['penalty']*360)/60)
        plr['Congestion Percentage'] = plr['penalty'] / plr['length']
        plr['length normalised'] = (plr['best travel time']-plr['best travel time'].min())/(plr['best travel time'].max()-plr['best travel time'].min())
        
        plr.plot.scatter(x = 'length normalised', y = 'Congestion Percentage')
        
        print('Alg : ' + str(experiment_labels[alg[0]][4:]) + '. Congestion Level : ' + str(congestion_labels_porto[exp][4:]) + '. Correlation Coefficient : ' + str(plr['length normalised'].corr(plr['Congestion Percentage'])))
        
#%% New York

ny_experiments = ['1','2','3','4']
algs = [['Naive Dijkstra','Time'],['Collective Top k','Time'],['A-Star','Time then Pen Desc'],['Coll A-Star','1']]

for exp in ny_experiments:
    for alg in algs:

        plr = pd.read_csv('/.../NY/Experiment_'+str(exp)+'/Results/'+str(alg[0])+'/'+str(alg[1])+'/Path Level Results - Exp '+str(exp)+'.csv')
        plr['Actual Penalty'] = round((plr['penalty']*360)/60)
        plr['Congestion Percentage'] = plr['penalty'] / plr['length']
        plr['length normalised'] = (plr['best travel time']-plr['best travel time'].min())/(plr['best travel time'].max()-plr['best travel time'].min())
        
        plr.plot.scatter(x = 'length normalised', y = 'Congestion Percentage')
        
        print('Alg : ' + str(experiment_labels[alg[0]][4:]) + '. Congestion Level : ' + str(congestion_labels_ny[exp][4:]) + '. Correlation Coefficient : ' + str(plr['length normalised'].corr(plr['Congestion Percentage'])))
        
#%% Uncontrolled Load Experiments


#%% Porto

world = 'Porto'

#Read in data

results = pd.read_csv('/.../'+str(world)+'/Full Results.csv')

#Drop IPR Heuristic

results = results[results['Algorithm'] != 'Impactful Path Replacement Penalty Heuristic']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.1']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.01']
results = results[results['Algorithm'] != 'Load Aware Dijkstra']

#Add Query Set Size Label

# results['query_set_size'] = results['Experiment'].str[:1].map(query_set_dict)

results['Query Set Size'] = results['Count Quickest Paths']  + results['Count Non Quickest Paths']

# Derive Fields

# Actual Travel Time

results['Actual Travel Time'] = results['Total Travel Time'] * 360

# Actual Congestion Penalty

results['Actual Congestion Penalty'] = results['Total Congestion Penalty'] * 360

# Average Journey Time

results['Average Journey Time'] = (results['Actual Travel Time'] / results['Query Set Size']) / 60

# Average Congstion Penalty

results['Congestion Flow Percent'] = (results['Actual Congestion Penalty'] / results['Actual Travel Time']) * 100

#ReLabel Experiment Names

results['Algorithm Label'] = results['Algorithm'].map(experiment_labels)

#Add Label for congestion level

results['Congestion Level'] = results['Experiment'].str[:1].map(congestion_labels_porto)

# Select only experiments 1,2,8,9

results = results[results['Experiment'].str[:1].isin(['1','2','8','9'])]

fig, ax = plt.subplots(1,3,figsize=(25,4))

fig_num = 0

for exp in ['8','9','1']:
    
    print('Experiment : ' + str(exp))
    print()
    
    nd = results[(results['Algorithm Label'] == '01. ND') & (results['Sort'] == 'Time') & (results['Experiment'].str[:1] == exp)]
    nd['Control Factor'] = nd['Experiment'].str[2:]
    nd.loc[nd['Control Factor'] == "", 'Control Factor'] = '1.0'
    nd = nd.pivot('Control Factor','Algorithm Label','Average Journey Time')
    nd.columns = ['ND']
    nd.index = nd.index.map(float) * 100
    print('ND : ')
    print(nd)
    print()
    
    TLATk = results[(results['Algorithm Label'] == '02. TLATk') & (results['Sort'] == 'Time') & (results['Experiment'].str[:1] == exp)]
    TLATk['Control Factor'] = TLATk['Experiment'].str[2:]
    TLATk.loc[TLATk['Control Factor'] == "", 'Control Factor'] = '1.0'
    TLATk = TLATk.pivot('Control Factor','Algorithm Label','Average Journey Time')
    TLATk.columns = ['TLATk']
    TLATk.index = TLATk.index.map(float) * 100
    print('TLATk : ')
    print(TLATk)
    print()
    
    laa_time = results[(results['Algorithm Label'] == '03. TLAA*') & (results['Sort'] == 'Time') & (results['Experiment'].str[:1] == exp)]
    laa_time['Control Factor'] = laa_time['Experiment'].str[2:]
    laa_time.loc[laa_time['Control Factor'] == "", 'Control Factor'] = '1.0'
    laa_time = laa_time.pivot('Control Factor','Algorithm Label','Average Journey Time')
    laa_time.columns = ['TLAA*']
    laa_time.index = laa_time.index.map(float) * 100
    print('laa_time : ')
    print(laa_time)
    print()
    
    coll_la = results[(results['Algorithm Label'] == '04. CS-MAT') & (results['Sort'] == 'No Sort') & (results['Experiment'].str[:1] == exp)]
    coll_la['Control Factor'] = coll_la['Experiment'].str[2:]
    coll_la.loc[coll_la['Control Factor'] == "", 'Control Factor'] = '1.0'
    coll_la = coll_la.pivot('Control Factor','Algorithm Label','Average Journey Time')
    coll_la.columns = ['CS-MAT']
    coll_la.index = coll_la.index.map(float) * 100
    print('coll_la : ')
    print(coll_la)
    print()
    print()
    
    
    nd.plot(ax = ax[fig_num], marker = nd_params[1], color = nd_params[0])
    TLATk.plot(ax = ax[fig_num], marker = tlatk_params[1], color = tlatk_params[0])
    laa_time.plot(ax = ax[fig_num], marker = tlaa_params[1], color = tlaa_params[0])
    coll_la.plot(ax = ax[fig_num], marker = cs_mat_params[1], color = cs_mat_params[0])
    ax[fig_num].set_xlabel('Control Factor (%)')
    if fig_num == 0:
        ax[fig_num].set_ylabel('AJT (Mins)')
    
    if fig_num == 0:
        ax[fig_num].set_title('2,000 Queries')
    elif fig_num == 1:
        ax[fig_num].set_title('5,000 Queries')
    elif fig_num == 2:
        ax[fig_num].set_title('10,000 Queries')
        
    if fig_num != 2:
        ax[fig_num].get_legend().remove()
    else:
        ax[fig_num].legend(labelspacing=.1)        

    fig_num += 1

fig.savefig(output_path+'Porto_Uncontrolled_Load.pdf', bbox_inches = "tight")
plt.show()

#%% New York

world = 'NY'

#Read in data

results = pd.read_csv('/.../'+str(world)+'/Full Results.csv')

#Drop IPR Heuristic

results = results[results['Algorithm'] != 'Impactful Path Replacement Penalty Heuristic']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.1']
results = results[results['Algorithm'] != 'Impactful Path Replacement Stop Factor 0.01']
results = results[results['Algorithm'] != 'Load Aware Dijkstra']

#Add Query Set Size Label

# results['query_set_size'] = results['Experiment'].str[:1].map(query_set_dict)

results['Query Set Size'] = results['Count Quickest Paths']  + results['Count Non Quickest Paths']

# Derive Fields

# Actual Travel Time

results['Actual Travel Time'] = results['Total Travel Time'] * 360

# Actual Congestion Penalty

results['Actual Congestion Penalty'] = results['Total Congestion Penalty'] * 360

# Average Journey Time

results['Average Journey Time'] = (results['Actual Travel Time'] / results['Query Set Size']) / 60

# Average Congstion Penalty

results['Congestion Flow Percent'] = (results['Actual Congestion Penalty'] / results['Actual Travel Time']) * 100

#ReLabel Experiment Names

results['Algorithm Label'] = results['Algorithm'].map(experiment_labels)

#Add Label for congestion level

results['Congestion Level'] = results['Experiment'].str[:1].map(congestion_labels_porto)

# Select only experiments 1,2,8,9

results = results[results['Experiment'].str[:1].isin(['1','2','3','4'])]

fig, ax = plt.subplots(1,3,figsize=(25,4))

fig_num = 0

for exp in ['1','2','3']:

    print('Experiment : ' + str(exp))
    print()    

    nd = results[(results['Algorithm Label'] == '01. ND') & (results['Sort'] == 'Time') & (results['Experiment'].str[:1] == exp)]
    nd['Control Factor'] = nd['Experiment'].str[2:]
    nd.loc[nd['Control Factor'] == "", 'Control Factor'] = '1.0'
    nd = nd.pivot('Control Factor','Algorithm Label','Average Journey Time')
    nd.columns = ['ND']
    nd.index = nd.index.map(float) * 100 
    print('ND : ')
    print(nd)
    print()
    
    TLATk = results[(results['Algorithm Label'] == '02. TLATk') & (results['Sort'] == 'Time') & (results['Experiment'].str[:1] == exp)]
    TLATk['Control Factor'] = TLATk['Experiment'].str[2:]
    TLATk.loc[TLATk['Control Factor'] == "", 'Control Factor'] = '1.0'
    TLATk = TLATk.pivot('Control Factor','Algorithm Label','Average Journey Time')
    TLATk.columns = ['TLATk']
    TLATk.index = TLATk.index.map(float) * 100
    print('TLATk : ')
    print(TLATk)
    print()
    
    laa_time = results[(results['Algorithm Label'] == '03. TLAA*') & (results['Sort'] == 'Time') & (results['Experiment'].str[:1] == exp)]
    laa_time['Control Factor'] = laa_time['Experiment'].str[2:]
    laa_time.loc[laa_time['Control Factor'] == "", 'Control Factor'] = '1.0'
    laa_time = laa_time.pivot('Control Factor','Algorithm Label','Average Journey Time')
    laa_time.columns = ['TLAA*']
    laa_time.index = laa_time.index.map(float) * 100
    print('laa_time : ')
    print(laa_time)
    print()
    
    coll_la = results[(results['Algorithm Label'] == '04. CS-MAT') & (results['Sort'] == 'No Sort') & (results['Experiment'].str[:1] == exp)]
    coll_la['Control Factor'] = coll_la['Experiment'].str[2:]
    coll_la.loc[coll_la['Control Factor'] == "", 'Control Factor'] = '1.0'
    coll_la = coll_la.pivot('Control Factor','Algorithm Label','Average Journey Time')
    coll_la.columns = ['CS-MAT']
    coll_la.index = coll_la.index.map(float) * 100
    print('coll_la : ')
    print(coll_la)
    print()
    print()
    
    nd.plot(ax = ax[fig_num], marker = nd_params[1], color = nd_params[0])
    TLATk.plot(ax = ax[fig_num], marker = tlatk_params[1], color = tlatk_params[0])
    laa_time.plot(ax = ax[fig_num], marker = tlaa_params[1], color = tlaa_params[0])
    coll_la.plot(ax = ax[fig_num], marker = cs_mat_params[1], color = cs_mat_params[0])
    ax[fig_num].set_xlabel('Control Factor (%)')
    if fig_num == 0:
        ax[fig_num].set_ylabel('AJT (Mins)')
    
    if fig_num == 0:
        ax[fig_num].set_title('10,000 Queries')
    elif fig_num == 1:
        ax[fig_num].set_title('25,000 Queries')
    elif fig_num == 2:
        ax[fig_num].set_title('50,000 Queries')
    
    if fig_num != 2:
        ax[fig_num].get_legend().remove()
    else:
        ax[fig_num].legend(labelspacing=.1)
        
    fig_num += 1
fig.savefig(output_path+'NY_Uncontrolled_Load.pdf', bbox_inches = "tight")
plt.show()