'''
01.2 Real World Set Up

Purpose of code: Read in real world raw data and process it such that is in correct format for experiments.
Specifically - reformat grid, derive edge level features, select queies subset for experimetns and depsosit data in correct folder stuctures for subsequent experiments to pick up
'''

import os
import pickle
import osmnx as ox
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

#Repository Path
path = ""
os.chdir(path)
#Path at which to land NY data
NY_folder = ""
#Path at which to land Porto data
porto_folder = ""
#Path where raw data stored
data_path = ""

from csp_toolset import *

parameters = pd.read_csv('parameters.csv',index_col = 0).to_dict()
t_dom = parameters['Value']['t_dom']
t_max = int(((1/t_dom) * 5 -1))
query_factor = int(t_dom * 3600)

query_set_sizes = [10000,25000,50000,100000,150000,250000,500000]

#%%

def CountFrequency(my_list): 
  
    # Creating an empty dictionary  
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
  
    nodes_to_print = []
    
    for key, value in freq.items(): 
        if value > 600:
            nodes_to_print.append(key)
            
    return nodes_to_print

#%% Read in New York

with open(data_path+"nyc-pickle.txt", "rb") as input_file:
    G = pickle.load(input_file)

edge_attributes = pd.read_csv(data_path+'nyc-edge-info.csv', index_col = [0,1])

road_multi_classes = []

graph_attributes = pd.DataFrame(columns = ['u-node','v-node','Length','Road Type','Oneway','Lanes','Speed Limit','Free Flow Max','Travel Time'])

i = 0
for node1, node2, data in G.edges(data=True):

    try:
        data['length'] = edge_attributes.xs((node1,node2))['length'].values[0]
        data['speed_limit'] = edge_attributes.xs((node1,node2))['speed_limit'].values[0]
        data['lanes'] = edge_attributes.xs((node1,node2))['lanes'].values[0]
        
        if edge_attributes.xs((node1,node2))['speed_limit'].values[0] > 20:
            headway = 5
            edge_length  = data['length'] = edge_attributes.xs((node1,node2))['length'].values[0]
            speed_limit = edge_attributes.xs((node1,node2))['speed_limit'].values[0]
            lanes = data['lanes'] = edge_attributes.xs((node1,node2))['lanes'].values[0]
        else:
            headway = 4
            edge_length  = data['length'] = edge_attributes.xs((node1,node2))['length'].values[0]
            speed_limit = edge_attributes.xs((node1,node2))['speed_limit'].values[0]
            lanes = data['lanes'] = edge_attributes.xs((node1,node2))['lanes'].values[0]
        
        data['ff_travel_time'] = round(((edge_length / speed_limit) / query_factor) ,4)
        ff_travel_time = round(((edge_length / speed_limit) / query_factor) ,4)
        data['ff_max'] = math.ceil(((edge_length / (speed_limit * headway)) + (headway+((ff_travel_time * 360)/2))) * lanes)
        
        print(data['ff_max'])
        
    except:
        data['length'] = edge_attributes.xs((node2,node1))['length'].values[0]
        data['speed_limit'] = edge_attributes.xs((node2,node1))['speed_limit'].values[0]
        data['lanes'] = edge_attributes.xs((node2,node1))['lanes'].values[0]

        if edge_attributes.xs((node2,node1))['speed_limit'].values[0] > 20:
            headway = 5
            edge_length  = data['length'] = edge_attributes.xs((node2,node1))['length'].values[0]
            speed_limit = edge_attributes.xs((node2,node1))['speed_limit'].values[0]
            lanes = data['lanes'] = edge_attributes.xs((node2,node1))['lanes'].values[0]
        else:
            headway = 4
            edge_length  = data['length'] = edge_attributes.xs((node2,node1))['length'].values[0]
            speed_limit = edge_attributes.xs((node2,node1))['speed_limit'].values[0]
            lanes = data['lanes'] = edge_attributes.xs((node2,node1))['lanes'].values[0]
        
        data['ff_travel_time'] = round(((edge_length / speed_limit) / query_factor) ,4)   
        ff_travel_time = round(((edge_length / speed_limit) / query_factor) ,4)
        data['ff_max'] = math.ceil(((edge_length / (speed_limit * headway)) + (headway+((ff_travel_time * 360)/2))) * lanes)
        
        print(data['ff_max'])
             
    #Collect data in dataframe
    
    graph_attributes.loc[i] = 0
    graph_attributes.loc[i]['u-node'] = node1
    graph_attributes.loc[i]['v-node'] = node2
    graph_attributes.loc[i]['Length'] = data['length']
    #Add Road Type - where list take second item as first item always "unclassified" (in NY data)
    if isinstance(data['highway'], list):
        graph_attributes.loc[i]['Road Type'] = data['highway'][1]
    else:
        graph_attributes.loc[i]['Road Type'] = data['highway']
    
    graph_attributes.loc[i]['Oneway'] = data['oneway']
    graph_attributes.loc[i]['Speed Limit'] = data['speed_limit']
    graph_attributes.loc[i]['Lanes'] = data['lanes']
    graph_attributes.loc[i]['Free Flow Max'] = data['ff_max']
    graph_attributes.loc[i]['Travel Time'] = data['ff_travel_time']
        
    i += 1

graph_attributes['Length'] = graph_attributes['Length'].astype(float)
graph_attributes['Lanes'] = graph_attributes['Lanes'].astype(int)
graph_attributes['Speed Limit'] = graph_attributes['Speed Limit'].astype(float)
graph_attributes['Free Flow Max'] = graph_attributes['Free Flow Max'].astype(int)
graph_attributes['Travel Time'] = graph_attributes['Travel Time'].astype(float)

node_mapping = {}

for i in G.nodes:
    node_mapping[i] = tuple([0,0,i])
    
G = nx.relabel_nodes(G, node_mapping)

#%%

'''
fig, ax = ox.plot_graph(G)
plt.tight_layout()

graph_attributes.hist(column='Length')
plt.show()
ax = sns.countplot(y="Road Type",data=graph_attributes)
plt.show()
ax = sns.countplot(y="Lanes",data=graph_attributes)
plt.show()
ax = sns.countplot(y="Speed Limit",data=graph_attributes)
plt.show()
ax = sns.countplot(y="Road Type",hue = 'Speed Limit',data=graph_attributes)
plt.show()
graph_attributes.hist(column='Free Flow Max')
plt.show()
graph_attributes.to_csv(data_path+'graph_attributes.csv')
plt.show()
graph_attributes.hist(column='Free Flow Max')
plt.show()
graph_attributes.hist(column='Travel Time')
plt.show()
'''

#%% Identify and delete disconnected nodes

i = 0
infeasible_routes = []
routes_calculated = 0
source_nodes_infeasible = []
target_nodes_infeasible = []
for source in G.nodes():
    print(i)
    i += 1
    for target in G.nodes():
        if source != target:
            path_exists = nx.has_path(G, source, target)
            routes_calculated += 1
            if path_exists:
                pass
            else:
                infeasible_routes.append([source, target])
                source_nodes_infeasible.append(source)
                target_nodes_infeasible.append(target)
        
disconnected_sources_nodes = CountFrequency(source_nodes_infeasible)
disconnected_target_nodes = CountFrequency(target_nodes_infeasible)

for n in set(disconnected_sources_nodes + disconnected_target_nodes):
    G.remove_node(n)

#%% Set up graph as di-graph

H = nx.DiGraph()

H.add_nodes_from(G.nodes)

for node1, node2, data in G.edges(data=True):
    H.add_edge(node1,node2,speed_limit = data['speed_limit'],length = data['length'],ff_max = data['ff_max'],ff_travel_time = data['ff_travel_time'])

#%% Read in Queries

queries_raw = pd.read_csv(data_path+"nyc-queries.csv")

from_nodes = []
to_nodes = []

i = 0

keep_index = []

for index,row in queries_raw.iterrows():
    print(i)
    orig_xy = (row['start_lat'], row['start_long'])
    target_xy = (row['end_lat'], row['end_long'])
    orig_node = ox.get_nearest_node(G, orig_xy, method='euclidean')
    target_node = ox.get_nearest_node(G, target_xy, method='euclidean')
    from_nodes.append(orig_node)
    to_nodes.append(target_node)
    if orig_node == target_node:
        keep_index.append(False)
    else:
        keep_index.append(True)
    i += 1

#%%

queries_raw['from'] = from_nodes
queries_raw['to'] = to_nodes
queries_raw = queries_raw[keep_index]
queries_first_4_hours = queries_raw[queries_raw['timeafterstart'] <= 14399]
queries_first_4_hours['t'] = round(queries_first_4_hours['timeafterstart'] / query_factor , 4)
queries_first_4_hours = queries_first_4_hours.drop(['start_long','start_lat','end_long','end_lat','duration','timeafterstart','start_nn','end_nn'], axis = 1)
queries_first_4_hours.columns = ['length','from','to','t']
queries = queries_first_4_hours[['from','to','t','length']]
queries['vehicle size'] = 1

#%% Oputput Data

#Set Up Folders

exp = 0
for q_size in query_set_sizes:
    exp += 1

    folder_name = NY_folder + 'Experiment_'+str(exp)
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    sub_folder_data = folder_name + '/Data'
    if not os.path.exists(sub_folder_data):
        os.makedirs(sub_folder_data)
        
    sub_folder_data_training = folder_name + '/Data/Training Data/'
    if not os.path.exists(sub_folder_data_training):
        os.makedirs(sub_folder_data_training)
    
    sub_folder_results = folder_name + '/Results'
    if not os.path.exists(sub_folder_results):
        os.makedirs(sub_folder_results)
    
    sub_folder_learning = folder_name + '/Learning'
    if not os.path.exists(sub_folder_learning):
        os.makedirs(sub_folder_learning)
    
    pickle.dump(H, open(sub_folder_data+'/graph_pickle.txt', 'wb'))
    
    queries.sample(n = q_size,replace = True,random_state=1).to_csv(sub_folder_data+'/queries.csv')
    
    for i in range(1,11):
        queries.sample(n = q_size,replace = True,random_state=1).to_csv(folder_name + '/Data/Training Data/queries_'+str(i)+'.csv')
    
    file = open(folder_name+'/experiment_details.txt','w')
    file.write('Experiment number ' + str(exp) + ' details : '+ '\n')
    file.write('Query set of size : ' + str(q_size) + '\n')
    file.write('Max time period : ' + str(t_dom) + '\n')
    file.close()

#%% Read in Porto

with open(data_path+"porto-pickle.txt", "rb") as input_file:
    G = pickle.load(input_file)

edge_attributes = pd.read_csv(data_path+'porto-edge-info.csv', index_col = [0,1])

road_multi_classes = []

graph_attributes = pd.DataFrame(columns = ['u-node','v-node','Length','Road Type','Oneway','Lanes','Speed Limit','Free Flow Max','Travel Time'])

#%%

i = 0

for node1, node2, data in G.edges(data=True):

    try:
        data['length'] = edge_attributes.xs((node1,node2))['length'].values[0]
        data['speed_limit'] = edge_attributes.xs((node1,node2))['speed_limit'].values[0]
        data['lanes'] = edge_attributes.xs((node1,node2))['lanes'].values[0]
        
        if edge_attributes.xs((node1,node2))['speed_limit'].values[0] > 20:
            headway = 5
            edge_length  = data['length'] = edge_attributes.xs((node1,node2))['length'].values[0]
            speed_limit = edge_attributes.xs((node1,node2))['speed_limit'].values[0]
            lanes = data['lanes'] = edge_attributes.xs((node1,node2))['lanes'].values[0]
        else:
            headway = 4
            edge_length  = data['length'] = edge_attributes.xs((node1,node2))['length'].values[0]
            speed_limit = edge_attributes.xs((node1,node2))['speed_limit'].values[0]
            lanes = data['lanes'] = edge_attributes.xs((node1,node2))['lanes'].values[0]
        
        data['ff_travel_time'] = round(((edge_length / speed_limit) / query_factor) ,4)
        ff_travel_time = round(((edge_length / speed_limit) / query_factor) ,4)
        data['ff_max'] = math.ceil(((edge_length / (speed_limit * headway)) + (headway+((ff_travel_time * 360)/2))) * lanes)
        
    except:
        data['length'] = edge_attributes.xs((node2,node1))['length'].values[0]
        data['speed_limit'] = edge_attributes.xs((node2,node1))['speed_limit'].values[0]
        data['lanes'] = edge_attributes.xs((node2,node1))['lanes'].values[0]

        if edge_attributes.xs((node2,node1))['speed_limit'].values[0] > 20:
            headway = 5
            edge_length  = data['length'] = edge_attributes.xs((node2,node1))['length'].values[0]
            speed_limit = edge_attributes.xs((node2,node1))['speed_limit'].values[0]
            lanes = data['lanes'] = edge_attributes.xs((node2,node1))['lanes'].values[0]
        else:
            headway = 4
            edge_length  = data['length'] = edge_attributes.xs((node2,node1))['length'].values[0]
            speed_limit = edge_attributes.xs((node2,node1))['speed_limit'].values[0]
            lanes = data['lanes'] = edge_attributes.xs((node2,node1))['lanes'].values[0]
        
        data['ff_travel_time'] = round(((edge_length / speed_limit) / query_factor) ,4)
        ff_travel_time = round(((edge_length / speed_limit) / query_factor) ,4)
        data['ff_max'] = math.ceil(((edge_length / (speed_limit * headway)) + (headway+((ff_travel_time * 360)/2))) * lanes)
    
    graph_attributes.loc[i] = 0
    graph_attributes.loc[i]['u-node'] = node1
    graph_attributes.loc[i]['v-node'] = node2
    graph_attributes.loc[i]['Length'] = data['length']
    
    if isinstance(data['highway'], list):
        if data['highway'][0] == 'unclassified':
            graph_attributes.loc[i]['Road Type'] = data['highway'][1]
        else:
            graph_attributes.loc[i]['Road Type'] = data['highway'][0]
    else:
        graph_attributes.loc[i]['Road Type'] = data['highway']
    graph_attributes.loc[i]['Oneway'] = data['oneway']
    graph_attributes.loc[i]['Speed Limit'] = data['speed_limit']
    graph_attributes.loc[i]['Lanes'] = data['lanes']
    graph_attributes.loc[i]['Free Flow Max'] = data['ff_max']
    graph_attributes.loc[i]['Travel Time'] = data['ff_travel_time']
    i += 1

graph_attributes['Length'] = graph_attributes['Length'].astype(float)
graph_attributes['Lanes'] = graph_attributes['Lanes'].astype(int)
graph_attributes['Speed Limit'] = graph_attributes['Speed Limit'].astype(float)
graph_attributes['Free Flow Max'] = graph_attributes['Free Flow Max'].astype(int)
graph_attributes['Travel Time'] = graph_attributes['Travel Time'].astype(float)

#%%

'''
fig, ax = ox.plot_graph(G)
plt.tight_layout()

graph_attributes.hist(column='Length')
plt.show()
ax = sns.countplot(y="Road Type",data=graph_attributes)
plt.show()
ax = sns.countplot(y="Lanes",data=graph_attributes)
plt.show()
ax = sns.countplot(y="Speed Limit",data=graph_attributes)
plt.show()
ax = sns.countplot(y="Road Type",hue = 'Speed Limit',data=graph_attributes)
plt.show()
graph_attributes.hist(column='Free Flow Max')
plt.show()
graph_attributes.to_csv(data_path+'graph_attributes.csv')
plt.show()
graph_attributes.hist(column='Free Flow Max')
plt.show()
graph_attributes.hist(column='Travel Time')
plt.show()
'''
#%%

node_mapping = {}

for i in G.nodes:
    node_mapping[i] = tuple([0,0,i])
    
G = nx.relabel_nodes(G, node_mapping)

#%% Identify and delete disconnected nodes

i = 0
infeasible_routes = []
routes_calculated = 0
source_nodes_infeasible = []
target_nodes_infeasible = []

for source in G.nodes():
    print(i)
    i += 1
    for target in G.nodes():
        if source != target:
            path_exists = nx.has_path(G, source, target)
            routes_calculated += 1
            if path_exists:
                pass
            else:
                infeasible_routes.append([source, target])
                source_nodes_infeasible.append(source)
                target_nodes_infeasible.append(target)
        
disconnected_sources_nodes = CountFrequency(source_nodes_infeasible)
disconnected_target_nodes = CountFrequency(target_nodes_infeasible)

for n in set(disconnected_sources_nodes + disconnected_target_nodes):
    G.remove_node(n)

#%% Set up graph as di-graph

H = nx.DiGraph()

H.add_nodes_from(G.nodes)

for node1, node2, data in G.edges(data=True):
    H.add_edge(node1,node2,speed_limit = data['speed_limit'],length = data['length'],ff_max = data['ff_max'],ff_travel_time = data['ff_travel_time'])

#%% Read in Queries

queries_raw = pd.read_csv(data_path+"porto-queries.csv")

from_nodes = []
to_nodes = []

keep_index = []

for index,row in queries_raw.iterrows():
    print(i)
    orig_xy = (row['start_lat'], row['start_long'])
    target_xy = (row['end_lat'], row['end_long'])
    orig_node = ox.get_nearest_node(G, orig_xy, method='euclidean')
    target_node = ox.get_nearest_node(G, target_xy, method='euclidean')
    from_nodes.append(orig_node)
    to_nodes.append(target_node)
    if orig_node == target_node:
        keep_index.append(False)
    else:
        keep_index.append(True)
    i += 1
    
#%%
queries_raw['from'] = from_nodes
queries_raw['to'] = to_nodes
queries_raw = queries_raw[keep_index]
queries_first_4_hours = queries_raw[queries_raw['timeafterstart'] <= 14399]
queries_first_4_hours['t'] = round(queries_first_4_hours['timeafterstart'] / query_factor,4)
queries_first_4_hours = queries_first_4_hours.drop(['start_long','start_lat','end_long','end_lat','duration','timeafterstart','start_nn','end_nn'], axis = 1)
queries_first_4_hours.columns = ['length','from','to','t']
queries = queries_first_4_hours[['from','to','t','length']]
queries['vehicle size'] = 1

#%% Oputput Data

#Set Up Folders
exp = 0
for q_size in query_set_sizes:
    exp += 1
    folder_name = porto_folder + 'Experiment_'+str(exp)
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    sub_folder_data = folder_name + '/Data'
    if not os.path.exists(sub_folder_data):
        os.makedirs(sub_folder_data)
        
    sub_folder_data_training = folder_name + '/Data/Training Data/'
    if not os.path.exists(sub_folder_data_training):
        os.makedirs(sub_folder_data_training)
    
    sub_folder_results = folder_name + '/Results'
    if not os.path.exists(sub_folder_results):
        os.makedirs(sub_folder_results)
    
    sub_folder_learning = folder_name + '/Learning'
    if not os.path.exists(sub_folder_learning):
        os.makedirs(sub_folder_learning)
    
    pickle.dump(H, open(sub_folder_data+'/graph_pickle.txt', 'wb'))
    
    queries.sample(n = q_size,replace = True,random_state=1).to_csv(sub_folder_data+'/queries.csv')
    
    for i in range(1,11):
        queries.sample(n = q_size,replace = True,random_state=1).to_csv(folder_name + '/Data/Training Data/queries_'+str(i)+'.csv')
    
    file = open(folder_name+'/experiment_details.txt','w')
    file.write('Experiment number ' + str(exp) + ' details : '+ '\n')
    file.write('Query set of size : ' + str(q_size) + '\n')
    file.write('Max time period : ' + str(t_dom) + '\n')
    file.close()