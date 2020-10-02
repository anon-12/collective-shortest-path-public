'''
3.1 - Online Learning

Purpose of code: Tests the query level algorithms across different sorting factors.

'''

#Import Modules
import os
import pandas as pd
import pickle
import ast
import time
import sys
import tensorflow as tf 
from keras.models import load_model
import h5py

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
#Path to output from training
train_data_path = exp_path + "/Learning/Training Data/"
#Path where high level results are captured
results_path = "/.../" + str(world) + "/"

G = pickle.load(open(exp_path + '/Data/graph_pickle.txt', 'rb'))
# load_predictor = tf.keras.models.load_model(exp_path+'/Learning/load_predictor.h5')

#Best paths

if world == 'NY' or world == 'Porto':
    node_to_node_best_paths = pd.read_csv(base_folder+'node_to_node_paths.csv', converters={"source": ast.literal_eval, "target": ast.literal_eval},index_col = [1,2])
    node_to_node_best_paths = node_to_node_best_paths.drop(node_to_node_best_paths.columns[0], axis=1)
    n2n_best_lengths = pd.read_csv(base_folder+'node_to_node_paths_lengths.csv', converters={"source": ast.literal_eval, "target": ast.literal_eval},index_col = [1,2])['path1']
else:
    node_to_node_best_paths = pd.read_csv(exp_path+'/Learning/node_to_node_paths.csv', converters={"source": ast.literal_eval, "target": ast.literal_eval},index_col = [1,2])
    node_to_node_best_paths = node_to_node_best_paths.drop(node_to_node_best_paths.columns[0], axis=1)
    n2n_best_lengths = pd.read_csv(exp_path+'/Learning/node_to_node_paths_lengths.csv', converters={"source": ast.literal_eval, "target": ast.literal_eval},index_col = [1,2])['path1']



#%% Set up Algorithms and results matrix

# algorithm_names = ['Capacity Aware Dijkstra - Random Query Sort','Capacity Aware Dijkstra - Sort Queries by t then dist','Capacity Aware Dijkstra - Sort Queries by dist then t','Collective Top K - Random Query Sort','Collective Top K - Sort Queries by t then dist','Collective Top K - Sort Queries by dist then t','Imapctful Path Replacement - Stop Ratio 10','Imapctful Path Replacement - Stop Ratio 5','Imapctful Path Replacement - Stop Ratio 1','A-St Shortest Path with Random Sort','IPR With Penalty Sort','A-St with Load Prediction Equal Weighting','A-St with Load Prediction Weighted For Actual Load','A-St with Load Prediction Weighted For Expected Load','A-St Shortest Path with Penalty Sort Descending','A-St Shortest Path with Penalty Sort Ascending']

system_level_results = pd.DataFrame(columns=['Algorithm','Sort','CPU Time', 'Total Travel Time','Count Quickest Paths','Count Non Quickest Paths','Total Congestion Penalty','Free Flow Edges Traversed','Congested Edges Traversed'])

zero_array = np.zeros(system_level_results.shape[1], dtype=int)

#%% Baseline 1 - Dijkstra no Sort

#Set up Row on System Level Results
alg_name = 'Naive Dijkstra'
sort_by = 'Random'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set

queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

if world != 'Synthetic':
    queries = queries.reset_index(drop = True)

#Path Level Results
path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])

cpu_time = 0

for index,row in queries.iterrows():
    cpu_start_time = time.time()
    
    best_path = nx.shortest_path(G, source=row['from'], target=row['to'], weight='length', method='dijkstra')
    
    cpu_end_time = time.time()
    
    #Record CPU Time
    cpu_time += (cpu_end_time - cpu_start_time)
    
    #Get Actual Length given Congestion

    visited = {}
    visited[row['from']] = 0
    num_con_flow_edges = 0
    t_start = row['t']
    congested_edges = []
    
    total_time_to_node = 0
    
    for j in range(0,len(best_path)-1):
        delay = False
        u_node = best_path[j]
        v_node = best_path[j+1]
        t_at_u = round(t_start + total_time_to_node,4)
        t_lower = int(t_at_u)
        
        edge_attributes = eam.xs((u_node,v_node))
        
        free_flow_max = edge_attributes['ff_max']
    
        #Get Load on Edge at t
        try:
            #Get Load in Time Step
            edge_load_at_t = elm.xs((u_node,v_node))[t_lower]
        except:
            edge_load_at_t = 0    
            
        #Get Delay exponent
        if edge_load_at_t <= free_flow_max:
            delay_exponent = 1
        else:
            delay_exponent = 1 / (edge_load_at_t - free_flow_max)
            delay = True
        
        #Arrival time at v
        t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge_attributes['ff_travel_time'],4)    
        
        #Total Travel time of Path to V
        total_time_to_node = t_at_v - t_start  
        
        # Travel time on Node
        tt_on_node = round(t_at_v - t_at_u,4)
    
        if delay == True:
            num_con_flow_edges += 1
            congested_edges.append([edge_attributes.name,tt_on_node])
            
        #Uppdate visisted matrix
        visited[v_node] = total_time_to_node
    
    path_length = round(total_time_to_node,4)    
    
    
    #Update Matrices
    
    cpu_start_time = time.time()
    
    elm = add_path_to_elm(row,best_path,visited,elm)
    epm = update_path_matrix(row,best_path,epm,index,visited)
    
    cpu_end_time = time.time()
    
    #Record CPU Time
    cpu_time += (cpu_end_time - cpu_start_time) 
    
    #Update Path Level Results

    path_level_results.loc[index]['path'] = best_path
    path_level_results.loc[index]['length'] = visited[row['to']]
    path_level_results.loc[index]['best path'] = best_path == best_path
    path_level_results.loc[index]['best travel time'] = get_tt_of_path(best_path,eam)
    path_level_results.loc[index]['number of edges'] = len(best_path)
    path_level_results.loc[index]['number of congested flow edges'] = num_con_flow_edges
    path_level_results.loc[index]['number of freeflow edges'] = len(best_path) - num_con_flow_edges
    path_level_results.loc[index]['penalty'] = round(visited[row['to']] - get_tt_of_path(best_path,eam),4)

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Baseline 2 - Dijkstra Sort by Time

#Set up Row on System Level Results
alg_name = 'Naive Dijkstra'
sort_by = 'Time'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int'])
queries = queries.drop(columns=['t_int'])

#Path Level Results
path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])

cpu_time = 0

for index,row in queries.iterrows():
    
    cpu_start_time = time.time()
    
    best_path = nx.shortest_path(G, source=row['from'], target=row['to'], weight='length', method='dijkstra')
    
    cpu_end_time = time.time()
    
    #Record CPU Time
    cpu_time += (cpu_end_time - cpu_start_time)
    
    #Get Actual Length given Congestion

    visited = {}
    visited[row['from']] = 0
    num_con_flow_edges = 0
    t_start = row['t']
    congested_edges = []
    
    total_time_to_node = 0
    
    for j in range(0,len(best_path)-1):
        delay = False
        u_node = best_path[j]
        v_node = best_path[j+1]
        t_at_u = round(t_start + total_time_to_node,4)
        t_lower = int(t_at_u)
        
        edge_attributes = eam.xs((u_node,v_node))
        
        free_flow_max = edge_attributes['ff_max']
    
        #Get Load on Edge at t
        try:
            #Get Load in Time Step
            edge_load_at_t = elm.xs((u_node,v_node))[t_lower]
        except:
            edge_load_at_t = 0    
            
        #Get Delay exponent
        if edge_load_at_t <= free_flow_max:
            delay_exponent = 1
        else:
            delay_exponent = 1 / (edge_load_at_t - free_flow_max)
            delay = True
        
        #Arrival time at v
        t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge_attributes['ff_travel_time'],4)    
        
        #Total Travel time of Path to V
        total_time_to_node = t_at_v - t_start  
        
        # Travel time on Node
        tt_on_node = round(t_at_v - t_at_u,4)
    
        if delay == True:
            num_con_flow_edges += 1
            congested_edges.append([edge_attributes.name,tt_on_node])
            
        #Uppdate visisted matrix
        visited[v_node] = total_time_to_node
    
    path_length = round(total_time_to_node,4)    
    
    
    #Update Matrices
    
    cpu_start_time = time.time()
    
    elm = add_path_to_elm(row,best_path,visited,elm)
    epm = update_path_matrix(row,best_path,epm,index,visited)
    
    cpu_end_time = time.time()
    
    #Record CPU Time
    cpu_time += (cpu_end_time - cpu_start_time) 
    
    #Update Path Level Results

    path_level_results.loc[index]['path'] = best_path
    path_level_results.loc[index]['length'] = visited[row['to']]
    path_level_results.loc[index]['best path'] = best_path == best_path
    path_level_results.loc[index]['best travel time'] = get_tt_of_path(best_path,eam)
    path_level_results.loc[index]['number of edges'] = len(best_path)
    path_level_results.loc[index]['number of congested flow edges'] = num_con_flow_edges
    path_level_results.loc[index]['number of freeflow edges'] = len(best_path) - num_con_flow_edges
    path_level_results.loc[index]['penalty'] = round(visited[row['to']] - get_tt_of_path(best_path,eam),4)

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 1 - Load Aware Dijkstr with Sort by Time

#Set up Row on System Level Results
alg_name = 'Load Aware Dijkstra'
sort_by = 'Time'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int'])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

#Run Algorithm

path_level_results, elm, epm = alg_LAD(queries,node_to_node_best_paths,G,elm,epm,eam)

cpu_end_time = time.time()

#Record CPU Time
cpu_time = cpu_end_time - cpu_start_time

#Output Results

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 2 - Load Aware Dijkstr with Sort by Time then Dist Ascending


#Set up Row on System Level Results
alg_name = 'Load Aware Dijkstra'
sort_by = 'Time then Dist Asc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

#Run Algorithm

path_level_results, elm, epm = alg_LAD(queries,node_to_node_best_paths,G,elm,epm,eam)

cpu_end_time = time.time()

#Record CPU Time
cpu_time = cpu_end_time - cpu_start_time

#Output Results

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 3 - Load Aware Dijkstr with Sort by Time then Dist Descending


#Set up Row on System Level Results
alg_name = 'Load Aware Dijkstra'
sort_by = 'Time then Dist Desc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

#Run Algorithm

path_level_results, elm, epm = alg_LAD(queries,node_to_node_best_paths,G,elm,epm,eam)

cpu_end_time = time.time()

#Record CPU Time
cpu_time = cpu_end_time - cpu_start_time

#Output Results

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)


#%% Test 4 - Load Aware Dijkstr with Sort by Time then Pen Ascending


#Set up Row on System Level Results
alg_name = 'Load Aware Dijkstra'
sort_by = 'Time then Pen Asc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

#Run Algorithm

path_level_results, elm, epm = alg_LAD(queries,node_to_node_best_paths,G,elm,epm,eam)

cpu_end_time = time.time()

#Record CPU Time
cpu_time = cpu_end_time - cpu_start_time

#Output Results

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 5 - Load Aware Dijkstr with Sort by Time then Pen Desc


#Set up Row on System Level Results
alg_name = 'Load Aware Dijkstra'
sort_by = 'Time then Pen Desc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

#Run Algorithm

path_level_results, elm, epm = alg_LAD(queries,node_to_node_best_paths,G,elm,epm,eam)

cpu_end_time = time.time()

#Record CPU Time
cpu_time = cpu_end_time - cpu_start_time

#Output Results

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 6 - Collective Top K with Sort by Time

#Set up Row on System Level Results
alg_name = 'Collective Top k'
sort_by = 'Time'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int'])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results, elm, epm = alg_collTopK(queries,node_to_node_best_paths,eam,elm,epm)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 7 - Collective Top K with Sort by Time then Dist Ascending

#Set up Row on System Level Results
alg_name = 'Collective Top k'
sort_by = 'Time then Dist Asc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results, elm, epm = alg_collTopK(queries,node_to_node_best_paths,eam,elm,epm)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 8 - Collective Top K with Sort by Time then Dist Descending

#Set up Row on System Level Results
alg_name = 'Collective Top k'
sort_by = 'Time then Dist Desc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results, elm, epm = alg_collTopK(queries,node_to_node_best_paths,eam,elm,epm)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 9 - Collective Top K with Sort by Time then Pen Ascending

#Set up Row on System Level Results
alg_name = 'Collective Top k'
sort_by = 'Time then Pen Asc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results, elm, epm = alg_collTopK(queries,node_to_node_best_paths,eam,elm,epm)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 10 - Collective Top K with Sort by Time then Pen Descending

#Set up Row on System Level Results
alg_name = 'Collective Top k'
sort_by = 'Time then Pen Desc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results, elm, epm = alg_collTopK(queries,node_to_node_best_paths,eam,elm,epm)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 11 - A-Star with Sort by Time

#Set up Row on System Level Results
alg_name = 'A-Star'
sort_by = 'Time'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int'])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results, elm, epm = alg_AStar(queries,node_to_node_best_paths,G,elm,epm,eam,n2n_best_lengths)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 12 - A-Star with Sort by Time then Dist Ascending

#Set up Row on System Level Results
alg_name = 'A-Star'
sort_by = 'Time then Dist Asc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results, elm, epm  = alg_AStar(queries,node_to_node_best_paths,G,elm,epm,eam,n2n_best_lengths)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 13 - A-Star with Sort by Time then Dist Descending

#Set up Row on System Level Results
alg_name = 'A-Star'
sort_by = 'Time then Dist Desc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results, elm, epm  = alg_AStar(queries,node_to_node_best_paths,G,elm,epm,eam,n2n_best_lengths)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 14 - A-Star with Sort by Time then Pen Ascending

#Set up Row on System Level Results
alg_name = 'A-Star'
sort_by = 'Time then Pen Asc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results, elm, epm  = alg_AStar(queries,node_to_node_best_paths,G,elm,epm,eam,n2n_best_lengths)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 15 - A-Star with Sort by Time then Pen Descending

#Set up Row on System Level Results
alg_name = 'A-Star'
sort_by = 'Time then Pen Desc'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results, elm, epm  = alg_AStar(queries,node_to_node_best_paths,G,elm,epm,eam,n2n_best_lengths)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 16 - A-Star With Medium Load Pred with Sort by Time
'''
#Set up Row on System Level Results
alg_name = 'A-Star Medium Load pred'
sort_by = 'Time'
pred_strength = 'Med'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int'])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 17 - A-Star With Medium Load Pred with Sort by Time then Dist Ascending

#Set up Row on System Level Results
alg_name = 'A-Star Medium Load pred'
sort_by = 'Time then Dist Asc'
pred_strength = 'Med'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 18 - A-Star With Medium Load Pred with Sort by Time then Dist Descending

#Set up Row on System Level Results
alg_name = 'A-Star Medium Load pred'
sort_by = 'Time then Dist Desc'
pred_strength = 'Med'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 19 - A-Star With Medium Load Pred with Sort by Time then Pen Ascending

#Set up Row on System Level Results
alg_name = 'A-Star Medium Load pred'
sort_by = 'Time then Pen Asc'
pred_strength = 'Med'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 20 - A-Star With Medium Load Pred with Sort by Time then Pen Descending

#Set up Row on System Level Results
alg_name = 'A-Star Medium Load pred'
sort_by = 'Time then Pen Desc'
pred_strength = 'Med'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)


#%% Test 21 - A-Star With Weak Load Pred with Sort by Time

#Set up Row on System Level Results
alg_name = 'A-Star Weak Load pred'
sort_by = 'Time'
pred_strength = 'Weak'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int'])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 22 - A-Star With Weak Load Pred with Sort by Time then Dist Ascending

#Set up Row on System Level Results
alg_name = 'A-Star Weak Load pred'
sort_by = 'Time then Dist Asc'
pred_strength = 'Weak'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 23 - A-Star With Weak Load Pred with Sort by Time then Dist Descending

#Set up Row on System Level Results
alg_name = 'A-Star Weak Load pred'
sort_by = 'Time then Dist Desc'
pred_strength = 'Weak'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 24 - A-Star With Weak Load Pred with Sort by Time then Pen Ascending

#Set up Row on System Level Results
alg_name = 'A-Star Weak Load pred'
sort_by = 'Time then Pen Asc'
pred_strength = 'Weak'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 25 - A-Star With Weak Load Pred with Sort by Time then Pen Descending

#Set up Row on System Level Results
alg_name = 'A-Star Weak Load pred'
sort_by = 'Time then Pen Desc'
pred_strength = 'Weak'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 26 - A-Star With Strong Load Pred with Sort by Time

#Set up Row on System Level Results
alg_name = 'A-Star Strong Load pred'
sort_by = 'Time'
pred_strength = 'Strong'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int'])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 27 - A-Star With Strong Load Pred with Sort by Time then Dist Ascending

#Set up Row on System Level Results
alg_name = 'A-Star Strong Load pred'
sort_by = 'Time then Dist Asc'
pred_strength = 'Strong'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 28 - A-Star With Strong Load Pred with Sort by Time then Dist Descending

#Set up Row on System Level Results
alg_name = 'A-Star Strong Load pred'
sort_by = 'Time then Dist Desc'
pred_strength = 'Strong'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','length'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 29 - A-Star With Strong Load Pred with Sort by Time then Pen Ascending

#Set up Row on System Level Results
alg_name = 'A-Star Strong Load pred'
sort_by = 'Time then Pen Asc'
pred_strength = 'Strong'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, True])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 30 - A-Star With Strong Load Pred with Sort by Time then Pen Descending

#Set up Row on System Level Results
alg_name = 'A-Star Strong Load pred'
sort_by = 'Time then Pen Desc'
pred_strength = 'Strong'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)

#Import Query Set
queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})

#Sort Query Set

queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int','Predicted Pen'], ascending = [True, False])
queries = queries.drop(columns=['t_int'])

cpu_start_time = time.time()

path_level_results,elm,epm = alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,load_predictor,n2n_best_lengths,pred_strength)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

'''
#%% Test 31 - Impactful Path Replacement - Stop Criteria as 0.1

alg_name = 'Impactful Path Replacement Stop Factor 0.1'
sort_by = 'Time'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by
stop_limit = 0.1

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)
#%Congest Edge Matrix
cem = edge_load_matrix(G)

queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int'])
queries = queries.drop(columns=['t_int'])

queries_handled = 0

cpu_start_time = time.time()

path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])
path_to_congested_edges = {}
# queries = queries.sort_values(by=['t','length'])

path_reassign_history = pd.DataFrame(columns=['path_id','original path','new path','time saving','Savings Ratio'])

for i in range(0,len(queries)):
# for i in range(0,int(len(queries)/2)):
    print(i)
    q = queries.iloc[i].tolist()
    q_id = queries.iloc[i].name
    
    initial_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
    initial_path = eval(initial_path)
    
    visited = {}
    visited[q[0]] = 0
    num_con_flow_edges = 0
    t_start = q[2]
    congested_edges = []
    
    total_time_to_node = 0
    
    for j in range(0,len(initial_path)-1):
        delay = False
        u_node = initial_path[j]
        v_node = initial_path[j+1]
        t_at_u = round(t_start + total_time_to_node,4)
        t_lower = int(t_at_u)
        
        edge_attributes = eam.xs((u_node,v_node))
        
        free_flow_max = edge_attributes['ff_max']
    
        #Get Load on Edge at t
        try:
            #Get Load in Time Step
            edge_load_at_t = elm.xs((u_node,v_node))[t_lower]
        except:
            edge_load_at_t = 0    
            
        #Get Delay exponent
        if edge_load_at_t <= free_flow_max:
            delay_exponent = 1
        else:
            delay_exponent = 1 / (edge_load_at_t - free_flow_max)
            delay = True
        
        #Arrival time at v
        t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge_attributes['ff_travel_time'],4)    
        
        #Total Travel time of Path to V
        total_time_to_node = t_at_v - q[2]    
        
        # Travel time on Node
        tt_on_node = round(t_at_v - t_at_u,4)
    
        if delay == True:
            num_con_flow_edges += 1
            congested_edges.append([edge_attributes.name,tt_on_node])
            
        #Uppdate visisted matrix
        visited[v_node] = total_time_to_node
    
    initial_length = round(total_time_to_node,4)

    elm = add_path_to_elm(q,initial_path,visited,elm)
    epm = update_path_matrix(q,initial_path,epm,q_id,visited)
    cem = add_path_cem(q,congested_edges,visited,cem)
    
    path_to_congested_edges[q_id] = [congested_edges,visited]
    
    path_level_results.loc[q_id]['path'] = initial_path
    path_level_results.loc[q_id]['length'] = visited[q[1]]
    path_level_results.loc[q_id]['best path'] = initial_path == initial_path
    path_level_results.loc[q_id]['best travel time'] = get_tt_of_path(initial_path,eam)
    path_level_results.loc[q_id]['number of edges'] = len(initial_path)
    path_level_results.loc[q_id]['number of congested flow edges'] = num_con_flow_edges
    path_level_results.loc[q_id]['number of freeflow edges'] = len(initial_path) - num_con_flow_edges
    path_level_results.loc[q_id]['penalty'] = round(visited[q[1]] - get_tt_of_path(initial_path,eam),4)

path_level_results,elm,epm,eam = ipr_iterate(stop_limit,cem,epm,elm,path_level_results,path_reassign_history,queries, eam,G,node_to_node_best_paths,path_to_congested_edges)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 32 - Impactful Path Replacement - Stop Criteria as 0.01

alg_name = 'Impactful Path Replacement Stop Factor 0.01'
sort_by = 'Time'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by
stop_limit = 0.01

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)
#%Congest Edge Matrix
cem = edge_load_matrix(G)

queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int'])
queries = queries.drop(columns=['t_int'])

queries_handled = 0

cpu_start_time = time.time()

path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])
path_to_congested_edges = {}
# queries = queries.sort_values(by=['t','length'])

path_reassign_history = pd.DataFrame(columns=['path_id','original path','new path','time saving','Savings Ratio'])

for i in range(0,len(queries)):
# for i in range(0,int(len(queries)/2)):
    print(i)
    q = queries.iloc[i].tolist()
    q_id = queries.iloc[i].name
    
    initial_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
    initial_path = eval(initial_path)
    
    visited = {}
    visited[q[0]] = 0
    num_con_flow_edges = 0
    t_start = q[2]
    congested_edges = []
    
    total_time_to_node = 0
    
    for j in range(0,len(initial_path)-1):
        delay = False
        u_node = initial_path[j]
        v_node = initial_path[j+1]
        t_at_u = round(t_start + total_time_to_node,4)
        t_lower = int(t_at_u)
        
        edge_attributes = eam.xs((u_node,v_node))
        
        free_flow_max = edge_attributes['ff_max']
    
        #Get Load on Edge at t
        try:
            #Get Load in Time Step
            edge_load_at_t = elm.xs((u_node,v_node))[t_lower]
        except:
            edge_load_at_t = 0    
            
        #Get Delay exponent
        if edge_load_at_t <= free_flow_max:
            delay_exponent = 1
        else:
            delay_exponent = 1 / (edge_load_at_t - free_flow_max)
            delay = True
        
        #Arrival time at v
        t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge_attributes['ff_travel_time'],4)    
        
        #Total Travel time of Path to V
        total_time_to_node = t_at_v - q[2]    
        
        # Travel time on Node
        tt_on_node = round(t_at_v - t_at_u,4)
    
        if delay == True:
            num_con_flow_edges += 1
            congested_edges.append([edge_attributes.name,tt_on_node])
            
        #Uppdate visisted matrix
        visited[v_node] = total_time_to_node
    
    initial_length = round(total_time_to_node,4)

    elm = add_path_to_elm(q,initial_path,visited,elm)
    epm = update_path_matrix(q,initial_path,epm,q_id,visited)
    cem = add_path_cem(q,congested_edges,visited,cem)
    
    path_to_congested_edges[q_id] = [congested_edges,visited]
    
    path_level_results.loc[q_id]['path'] = initial_path
    path_level_results.loc[q_id]['length'] = visited[q[1]]
    path_level_results.loc[q_id]['best path'] = initial_path == initial_path
    path_level_results.loc[q_id]['best travel time'] = get_tt_of_path(initial_path,eam)
    path_level_results.loc[q_id]['number of edges'] = len(initial_path)
    path_level_results.loc[q_id]['number of congested flow edges'] = num_con_flow_edges
    path_level_results.loc[q_id]['number of freeflow edges'] = len(initial_path) - num_con_flow_edges
    path_level_results.loc[q_id]['penalty'] = round(visited[q[1]] - get_tt_of_path(initial_path,eam),4)

path_level_results,elm,epm,eam = ipr_iterate(stop_limit,cem,epm,elm,path_level_results,path_reassign_history,queries, eam,G,node_to_node_best_paths,path_to_congested_edges)

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)

#%% Test 33 - IPR Targeting Highest Penalty Paths

alg_name = 'Impactful Path Replacement Penalty Heuristic'
sort_by = 'Time'
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

# Set up Matrices
# Capacity Time Marix  
eam = edge_attribute_matrix(G)
#Load Matrix
elm = edge_load_matrix(G)
#Path Matrix
epm = path_matrix(G)
#%Congest Edge Matrix
cem = edge_load_matrix(G)

queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
if world != 'Synthetic':
    queries = queries.reset_index(drop = True)
queries['t_int'] = queries['t'].astype(int)
queries = queries.sort_values(by=['t_int'])
queries = queries.drop(columns=['t_int'])

queries_handled = 0

cpu_start_time = time.time()

path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])
path_to_congested_edges = {}
# queries = queries.sort_values(by=['t','length'])

path_reassign_history = pd.DataFrame(columns=['path_id','original path','new path','time saving','Savings Ratio'])

for i in range(0,len(queries)):
# for i in range(0,int(len(queries)/2)):
    print(i)
    q = queries.iloc[i].tolist()
    q_id = queries.iloc[i].name
    
    initial_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
    initial_path = eval(initial_path)
    
    visited = {}
    visited[q[0]] = 0
    num_con_flow_edges = 0
    t_start = q[2]
    congested_edges = []
    
    total_time_to_node = 0
    
    for j in range(0,len(initial_path)-1):
        delay = False
        u_node = initial_path[j]
        v_node = initial_path[j+1]
        t_at_u = round(t_start + total_time_to_node,4)
        t_lower = int(t_at_u)
        
        edge_attributes = eam.xs((u_node,v_node))
        
        free_flow_max = edge_attributes['ff_max']
    
        #Get Load on Edge at t
        try:
            #Get Load in Time Step
            edge_load_at_t = elm.xs((u_node,v_node))[t_lower]
        except:
            edge_load_at_t = 0    
            
        #Get Delay exponent
        if edge_load_at_t <= free_flow_max:
            delay_exponent = 1
        else:
            delay_exponent = 1 / (edge_load_at_t - free_flow_max)
            delay = True
        
        #Arrival time at v
        t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge_attributes['ff_travel_time'],4)    
        
        #Total Travel time of Path to V
        total_time_to_node = t_at_v - q[2]    
        
        # Travel time on Node
        tt_on_node = round(t_at_v - t_at_u,4)
    
        if delay == True:
            num_con_flow_edges += 1
            congested_edges.append([edge_attributes.name,tt_on_node])
            
        #Uppdate visisted matrix
        visited[v_node] = total_time_to_node
    
    initial_length = round(total_time_to_node,4)

    elm = add_path_to_elm(q,initial_path,visited,elm)
    epm = update_path_matrix(q,initial_path,epm,q_id,visited)
    cem = add_path_cem(q,congested_edges,visited,cem)
    
    path_to_congested_edges[q_id] = [congested_edges,visited]
    
    path_level_results.loc[q_id]['path'] = initial_path
    path_level_results.loc[q_id]['length'] = visited[q[1]]
    path_level_results.loc[q_id]['best path'] = initial_path == initial_path
    path_level_results.loc[q_id]['best travel time'] = get_tt_of_path(initial_path,eam)
    path_level_results.loc[q_id]['number of edges'] = len(initial_path)
    path_level_results.loc[q_id]['number of congested flow edges'] = num_con_flow_edges
    path_level_results.loc[q_id]['number of freeflow edges'] = len(initial_path) - num_con_flow_edges
    path_level_results.loc[q_id]['penalty'] = round(visited[q[1]] - get_tt_of_path(initial_path,eam),4)
    
for i in range(1,11):
    lower_bound = int((path_level_results.shape[0]/10) * i) - 100
    upper_bound = int((path_level_results.shape[0]/10) * i)
    
    globals()['queries_tenth_'+str(i)] = path_level_results.iloc[lower_bound:upper_bound]
    
    globals()['queries_tenth_'+str(i)] = globals()['queries_tenth_'+str(i)].sort_values(by=['penalty'], ascending=False)
    
    for j in range(0,10):
        
        path_id_to_reassign = globals()['queries_tenth_'+str(i)].iloc[j].name
        print('i : ' + str(i))
        print('j : ' + str(j))    
        
        #Get original query
        q = queries.loc[path_id_to_reassign]
        #Identify actual best path
        q_best_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
        q_best_path = eval(q_best_path)
        
        #Mark originally assigned path and length
        original_path = path_level_results.loc[path_id_to_reassign]['path']
        original_length = path_level_results.loc[path_id_to_reassign]['length']
        
        #ELM
        elm = remove_path_to_elm(q,original_path,path_to_congested_edges[path_id_to_reassign][1],elm)
        #EPM    
        epm = remove_from_path_matrix(q,original_path,epm,path_id_to_reassign,path_to_congested_edges[path_id_to_reassign][1])
        #CEM
        cem = remove_path_cem(q,path_to_congested_edges[path_id_to_reassign][0],path_to_congested_edges[path_id_to_reassign][1],cem)
        
        #Reselect path using LAD
        visited,path,congested_flow_edges = capacity_aware_dijkstra(q,G,elm)
        
        path_t_saving = original_length - visited[q[1]]
        
        print('Path Savings : ' + str(path_t_saving))
        
        if path_t_saving > 0:
            #Benefit detected - update system and path level results with new paths
            visited = {}
            visited[q[0]] = 0
            num_con_flow_edges = 0
            t_start = q[2]
            total_time_to_node = 0
            
            for j in range(0,len(path)-1):
                u_node = path[j]
                v_node = path[j+1]
                t_at_u = round(t_start + total_time_to_node,4)
                t_lower = int(t_at_u)
                
                edge_attributes = eam.xs((u_node,v_node))
                
                free_flow_max = edge_attributes['ff_max']
            
                #Get Load on Edge at t
                try:
                    #Get Load in Time Step
                    edge_load_at_t = elm.xs((u_node,v_node))[t_lower]
                except:
                    edge_load_at_t = 0
                    
                #Get Delay exponent
                if edge_load_at_t <= free_flow_max:
                    delay_exponent = 1
                else:
                    delay_exponent = 1 / (edge_load_at_t - free_flow_max)
                    num_con_flow_edges += 1
                
                #Arrival time at v
                t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge_attributes['ff_travel_time'],4)    
                
                #Total Travel time of Path to V
                total_time_to_node = t_at_v - q[2]    
                
                # Travel time on Node
                tt_on_node = round(t_at_v - t_at_u,4)
            
                #Uppdate visisted matrix
                visited[v_node] = total_time_to_node
            
            actual_length = round(total_time_to_node,4)
            
            #Update system with new path
            
            #ELM
            elm = add_path_to_elm(q,path,visited,elm)
            #EPM
            epm = update_path_matrix(q,path,epm,path_id_to_reassign,visited)
            
            #Update Record Results
            path_level_results.loc[path_id_to_reassign]['path'] = path
            path_level_results.loc[path_id_to_reassign]['length'] = actual_length
            path_level_results.loc[path_id_to_reassign]['best path'] = path == q_best_path
            path_level_results.loc[path_id_to_reassign]['best travel time'] = get_tt_of_path(q_best_path,eam)
            path_level_results.loc[path_id_to_reassign]['number of edges'] = len(path)
            path_level_results.loc[path_id_to_reassign]['number of congested flow edges'] = num_con_flow_edges
            path_level_results.loc[path_id_to_reassign]['number of freeflow edges'] = len(path) - num_con_flow_edges
            path_level_results.loc[path_id_to_reassign]['penalty'] = round(actual_length - get_tt_of_path(q_best_path,eam),4)
        
        
        else:
            #No Bnefit - Re-Add Paths to System
            
            #ELM
            elm = add_path_to_elm(q,original_path,path_to_congested_edges[path_id_to_reassign][1],elm)
            #EPM
            epm = update_path_matrix(q,original_path,epm,path_id_to_reassign,path_to_congested_edges[path_id_to_reassign][1])
            #CEM
            cem = add_path_cem(q,path_to_congested_edges[path_id_to_reassign][0],path_to_congested_edges[path_id_to_reassign][1],cem)
        print()

cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)