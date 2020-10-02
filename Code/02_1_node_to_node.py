'''
2.1 Node to Node

About Code - Prepocessing of experimental data for CSP Experiments. Deriving n2n best paths and building classifiers
'''

#Import Modules
import os
import pandas as pd
import ast
import time
import pickle
import sys
from ast import literal_eval as make_tuple


#Experiment Parameters
exp = 5
world = 'NY'

# Read in paramters

parameters = pd.read_csv('parameters.csv',index_col = 0).to_dict()
t_dom = parameters['Value']['t_dom']
t_max = int(((1/t_dom) * 5 -1))
edge_weight = 'ff_travel_time'
file_exists = False

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



#%% Get Input Data

G = pickle.load(open(exp_path + '/Data/graph_pickle.txt', 'rb'))
eam = edge_attribute_matrix(G)

#%% Get top 5 paths and lengths node to node

if world == 'NY' or world == 'Porto':
    if os.path.exists(results_path+'node_to_node_paths.csv') and os.path.exists(results_path+'node_to_node_paths_lengths.csv'):
        file_exists = True
        node_to_node_best_paths = pd.read_csv(results_path+'node_to_node_paths.csv',usecols = [1,2,3,4,5,6,7], converters={"source": ast.literal_eval,"target": ast.literal_eval})
        node_to_node_best_paths_length = pd.read_csv(results_path+'node_to_node_paths_lengths.csv',usecols = [1,2,3,4,5,6,7], converters={"source": ast.literal_eval,"target": ast.literal_eval})

#%%
if file_exists == False:
    cpu_start_time = time.time()
    k = 5
    infeasible_routes = []
    
    shortest_paths = []
    lengths = []
    for source in G.nodes():
        for target in G.nodes():
            if source != target:
                sp_append = [source,target]
                length_append = [source,target]
                try:
                    for path in k_shortest_paths(G, source, target, k, edge_weight):
                        sp_append.append(path)
                        length_append.append(get_tt_of_path(path,eam))
                    shortest_paths.append(sp_append)
                    lengths.append(length_append)
                except:
                    infeasible_routes.append([source, target])
    if world == 'NY' or world == 'Porto':
        node_to_node_best_paths = pd.DataFrame(shortest_paths,columns = ['source','target','path1','path2','path3','path4','path5'])
        node_to_node_best_paths.to_csv(results_path+'node_to_node_paths.csv')
        
        node_to_node_best_paths_length = pd.DataFrame(lengths,columns = ['source','target','path1','path2','path3','path4','path5'])
        node_to_node_best_paths_length.to_csv(results_path+'node_to_node_paths_lengths.csv')        
    else:
        node_to_node_best_paths = pd.DataFrame(shortest_paths,columns = ['source','target','path1','path2','path3','path4','path5'])
        node_to_node_best_paths.to_csv(exp_path+'/Learning/node_to_node_paths.csv')
        
        node_to_node_best_paths_length = pd.DataFrame(lengths,columns = ['source','target','path1','path2','path3','path4','path5'])
        node_to_node_best_paths_length.to_csv(exp_path+'/Learning/node_to_node_paths_lengths.csv')
    
    cpu_end_time = time.time()
    
    node_to_node_processing_time_cpu = cpu_end_time - cpu_start_time

#%% Process Training Query Sets through Load Aware A* to get ELM and PLR for each

cpu_start_time = time.time()

#Get Matrices ready for pre-processing

#Reindex best paths on source and target
node_to_node_best_paths = node_to_node_best_paths.set_index(['source','target'])
#Reindex best length on source and target, and only keep best path
n2n_best_lengths = node_to_node_best_paths_length.set_index(['source','target'])['path1']

#%%
# Path Level Training Set

pl_training_list = []

#Iterate through training sets
for t in range(1,11):

    #Load Matrix
    elm = edge_load_matrix(G)
    #Path Matrix
    epm = path_matrix(G)
    
    print('----------------  Next Training Set : ' + str(t ))
    
    #Load query Set
    queries = pd.read_csv(train_queries_path+'queries_'+str(t)+'.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
    
    # path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])

    for i in range(0,len(queries)):
        q = queries.iloc[i].tolist()
        q_id = queries.iloc[i].name
        
        q_best_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
        
        if file_exists == True:

            q_best_path_tuples = []
            counter = 0
            for i in [x.strip() for x in q_best_path[1:-1].split(')')][:-1]:
                counter += 1
                if counter == 1:
                    new_string = i + ')'
                    q_best_path_tuples.append(make_tuple(new_string))
                else:
                    new_string = i[2:] + ')'
                    q_best_path_tuples.append(make_tuple(new_string))
            q_best_path = q_best_path_tuples            
        
        #Run Algorithm
        visited,path,congested_flow_edges = load_aware_a_star(q,G,elm,best_path,n2n_best_lengths)
        
        # Update Matrices
        elm = add_path_to_elm(q,path,visited,elm)
        epm = update_path_matrix(q,path,epm,q_id,visited)
        
        pl_training_list.append({'U Node' : 'u_' + str(q[0]),'V Node' : 'v_' + str(q[1]),'T':q[2],'Penalty' : round(visited[q[1]],2) - round(get_tt_of_path(q_best_path,eam),2)})
        
    output_training_results(None,elm,epm,exp_path,t)

plr_training = pd.DataFrame(pl_training_list)

plr_training.to_csv(train_data_path + 'path_level_training_data.csv')

cpu_end_time = time.time()

process_training_query_sets_cpu = cpu_end_time - cpu_start_time


#%%

metrics = pd.read_csv(results_path+'Offline Learning Metrics.csv',index_col = 0)
metrics.loc[exp] = 0
metrics.loc[exp]['Node to Node CPU'] = node_to_node_processing_time_cpu
metrics.loc[exp]['Process Training Data CPU'] = process_training_query_sets_cpu
metrics.to_csv(results_path+'Offline Learning Metrics.csv')