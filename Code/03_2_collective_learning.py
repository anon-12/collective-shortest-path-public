'''
3.2 Collective Learning

Implementation of Collective Search for Minimal Arrival Time
'''

#Import Modules
import os
import statistics
import math
import pandas as pd
import pickle
import ast
import time
import sys

#Experiment Parameters
exp = int(sys.argv[1])
world = str(sys.argv[2])
load_controlled = float(sys.argv[3])
load_uncontrolled = float(1 - load_controlled)
exp_tag = str(exp)+'_'+str(load_controlled)
print('Experiment : ' + str(exp))
print('World : ' + str(world))
print('Load Controlled : ' + str(load_controlled))
print('Load Uncontrolled : ' + str(load_uncontrolled))
print('Experiment Label for Output : ' + str(exp_tag))

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

#Best paths

if world == 'NY' or world == 'Porto':
    node_to_node_best_paths = pd.read_csv(base_folder+'node_to_node_paths.csv', converters={"source": ast.literal_eval, "target": ast.literal_eval},index_col = [1,2])
    node_to_node_best_paths = node_to_node_best_paths.drop(node_to_node_best_paths.columns[0], axis=1)
    n2n_best_lengths = pd.read_csv(base_folder+'node_to_node_paths_lengths.csv', converters={"source": ast.literal_eval, "target": ast.literal_eval},index_col = [1,2])['path1']
else:
    node_to_node_best_paths = pd.read_csv(exp_path+'/Learning/node_to_node_paths.csv', converters={"source": ast.literal_eval, "target": ast.literal_eval},index_col = [1,2])
    node_to_node_best_paths = node_to_node_best_paths.drop(node_to_node_best_paths.columns[0], axis=1)
    n2n_best_lengths = pd.read_csv(exp_path+'/Learning/node_to_node_paths_lengths.csv', converters={"source": ast.literal_eval, "target": ast.literal_eval},index_col = [1,2])['path1']


#%% ELM Set

if load_controlled == 1:
    elm_master = edge_load_matrix(G)
else:
    u_nodes_list = []
    v_nodes_list = []
    
    for edge in G.edges(data=True):
        u_nodes_list.append(edge[0])
        v_nodes_list.append(edge[1])
    
    cols = ['source','target'] + list(np.arange(t_max+1))
    edge_cap_matrix = pd.DataFrame(columns=cols)
    
    edge_cap_matrix['source'] = u_nodes_list
    edge_cap_matrix['target'] = v_nodes_list
    edge_cap_matrix = edge_cap_matrix.set_index(['source','target'])
    
    for index,row in edge_cap_matrix.iterrows():
        for i in range(1,11):
            elm = pd.read_csv(train_data_path+'elm_'+str(i)+'.csv',index_col = [0,1],converters={"source": ast.literal_eval,"target": ast.literal_eval})
            if i == 1:
                for j in range(0,50):
                    try:
                        row[j] = [int(elm.xs(index)[j])]
                    except:
                        row[j] = [0]
            else:
                for j in range(0,50):
                    try:
                        row[j].append(int(elm.xs(index)[j]))
                    except:
                        row[j].append(int(0))
                    
    elm_master = pd.DataFrame(columns=cols)
    
    elm_master['source'] = u_nodes_list
    elm_master['target'] = v_nodes_list
    elm_master = elm_master.set_index(['source','target'])
    
    for index,row in elm_master.iterrows():
        for i in range(0,50):
            row[i] = int(math.ceil(float(statistics.mean(edge_cap_matrix.xs(index)[i]) * load_uncontrolled)))

#%% Add expected arrival time to query set

if load_controlled == 1:
    queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
else:
    queries = pd.read_csv(exp_path+'/Data/queries_all.csv', index_col = 0, converters={"from": ast.literal_eval, "to": ast.literal_eval})
    queries = queries.sample(n = int(queries.shape[0] * load_controlled))
    queries = queries.reset_index(drop = True)

ff_arrival_time = []
ff_travel_time = []

for index,row in queries.iterrows():
    ff_arrival_time.append(row['t'] + n2n_best_lengths.xs((row['from'],row['to'])))
    ff_travel_time.append(n2n_best_lengths.xs((row['from'],row['to'])))

queries['ff_arrival_time'] = ff_arrival_time
queries['ff_travel_time'] = ff_travel_time
queries = queries.sort_values(by=['ff_arrival_time'])
queries = queries.reset_index(drop = True)

# Get average positive predicted penalty per tau
backup_predicted_pen = queries['Predicted Pen'].clip(lower=0).quantile(0.75)

# Set Up Matrices

#Path Level Results
path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])

# Set up Matrices
# Capacity Time Marix
eam = edge_attribute_matrix(G)
#Load Matrix
elm = elm_master.copy()
#Path Matrix
epm = path_matrix(G)

#%%

current_q_id = 0
query_count = queries.shape[0]-1

cpu_start_time = time.time()
re_check_q = False

while current_q_id <= query_count:

    print('Current Query To Check : ' + str(current_q_id))
    
    try:
        q_to_check = queries.loc[current_q_id]
    except:
        current_q_id += 1
        print('Skip iteration until find query')
        print()
        continue
    
    ff_best_path = eval(node_to_node_best_paths.xs((q_to_check['from'],q_to_check['to']))['path1'])
    
    visited = {}
    visited[q_to_check['from']] = 0
    t_start = q_to_check['t']
    
    total_time_to_node = 0
    
    congestion_encountered = False
    
    for j in range(0,len(ff_best_path)-1):
        u_node = ff_best_path[j]
        v_node = ff_best_path[j+1]
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
            congestion_encountered = True
            break
        
        #Arrival time at v
        t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge_attributes['ff_travel_time'],4)    
        
        #Total Travel time of Path to V
        total_time_to_node = t_at_v - t_start  
        
        # Travel time on Node
        tt_on_node = round(t_at_v - t_at_u,4)
            
        #Uppdate visisted matrix
        visited[v_node] = total_time_to_node
    
    path_length = round(total_time_to_node,4)
    
    if congestion_encountered == False:
        print('No Congestion')
        #Add path to system
        elm = add_path_to_elm(q_to_check,ff_best_path,visited,elm)
        epm = update_path_matrix(q_to_check,ff_best_path,epm,index,visited)
    
        path_level_results.loc[current_q_id]['path'] = ff_best_path
        path_level_results.loc[current_q_id]['length'] = visited[q_to_check['to']]
        path_level_results.loc[current_q_id]['best path'] = ff_best_path == ff_best_path
        path_level_results.loc[current_q_id]['best travel time'] = get_tt_of_path(ff_best_path,eam)
        path_level_results.loc[current_q_id]['number of edges'] = len(ff_best_path)
        path_level_results.loc[current_q_id]['number of congested flow edges'] = 0
        path_level_results.loc[current_q_id]['number of freeflow edges'] = len(ff_best_path)
        path_level_results.loc[current_q_id]['penalty'] = round(visited[q_to_check['to']] - get_tt_of_path(ff_best_path,eam),4)    
        
        print('Increment Query ID')
        current_q_id += 1
    
    else:
        #Run LAA* on x number of paths
        #Get all queries get current query
        
        print('Congestion Encountered')
        
        if q_to_check['Predicted Pen'] > 0:
            arrival_time_lim = q_to_check['ff_arrival_time'] + q_to_check['Predicted Pen']
        else:
            arrival_time_lim = q_to_check['ff_arrival_time'] + backup_predicted_pen
        
        #If first time processing query then set up mini EPM
        if re_check_q == False:
            print('Create Query EPM')
            q_epm = path_matrix(G)        
        
            queries_for_la = queries.loc[current_q_id:]
            queries_for_la = queries_for_la[queries_for_la['ff_arrival_time'] <= arrival_time_lim]
            
            print('Number of queries being checked : ' + str(queries_for_la.shape[0]))
             
            la_results = pd.DataFrame(index = queries_for_la.index,columns = ['Query','Q ID','Q Best Path','Visited','Path','Congested Edges','Load Aware Arrival Time'])
            intersecting_paths = list(queries_for_la.index)
        else:
        
            # Identfy Paths that Share edge with Q ID
            
            print('Identify Paths Itersecting with Query : ' + str(q_id))
            
            intersecting_paths = []
            
            for index,row in q_epm.iterrows():
                for s_index,value in row.items():
                    if type(value) is list:
                        if q_id in value:
                            intersecting_paths = intersecting_paths + value
                            
            intersecting_paths = list(set(intersecting_paths))
            
            
            # Remove Paths from EPM
            
            for p_id in intersecting_paths:
                p_q = la_results.loc[p_id]['Query']
                p_path = la_results.loc[p_id]['Path']
                p_visited = la_results.loc[p_id]['Visited']
                q_epm = remove_from_path_matrix(p_q,p_path,q_epm,p_id,p_visited)
                if p_id == q_id:
                    la_results = la_results.drop(q_id)
            
            intersecting_paths.remove(q_id)
            
            #Select paths for reassign
            queries_for_la = queries.loc[intersecting_paths]
            print('Number of queries being checked : ' + str(queries_for_la.shape[0]))
        
        for index_la, row_la in queries_for_la.iterrows():
            if index_la in intersecting_paths:
                print('Checking Path : ' + str(index_la))
                q_to_check = queries.loc[index_la].tolist()
                q_to_check_id = queries.loc[index_la].name
                
                q_to_check_best_path = node_to_node_best_paths.xs((q_to_check[0], q_to_check[1]))['path1']
                q_to_check_best_path = eval(q_to_check_best_path)
                
                la_results.loc[index_la]['Query'] = q_to_check
                la_results.loc[index_la]['Q ID'] = q_to_check_id
                la_results.loc[index_la]['Q Best Path'] = q_to_check_best_path
                
                visited,path,congested_flow_edges = load_aware_a_star(q_to_check,G,elm,best_path,n2n_best_lengths)
                #add path to q_elm
                q_epm = update_path_matrix(q_to_check,path,q_epm,q_to_check_id,visited)
                
                la_results.loc[index_la]['Visited'] = visited
                la_results.loc[index_la]['Path'] = path
                la_results.loc[index_la]['Congested Edges'] = congested_flow_edges
                la_results.loc[index_la]['Load Aware Arrival Time'] = round(row_la['t'] + visited[row_la['to']],4)
        
        la_results = la_results.sort_values(by=['Load Aware Arrival Time'])
        
        best_result = la_results.iloc[0]
        
        q = best_result['Query']
        q_id = best_result['Q ID']
        q_best_path = best_result['Q Best Path']
        visited = best_result['Visited']
        path = best_result['Path']
        congested_flow_edges = best_result['Congested Edges']
        
        print('Path Assigned : ' + str(q_id))
        if q_id == current_q_id:
            print('Increment Query ID')
            current_q_id += 1
            re_check_q = False
        else:
            print('Check Query Again : ' + str(current_q_id))
            queries = queries.drop(q_id)
            re_check_q = True
        
        # Update Matrices
        elm = add_path_to_elm(q,path,visited,elm)
        epm = update_path_matrix(q,path,epm,q_id,visited)
        
        #Record Results
        path_level_results.loc[q_id]['path'] = path
        path_level_results.loc[q_id]['length'] = visited[q[1]]
        path_level_results.loc[q_id]['best path'] = path == q_best_path
        path_level_results.loc[q_id]['best travel time'] = get_tt_of_path(q_best_path,eam)
        path_level_results.loc[q_id]['number of edges'] = len(path)
        #Get number of congested flow edges
        num_con_flow_edges = 0
        last_e = path[0]
        for e in path[1:]:
            edge_check = [last_e,e]
            if edge_check in congested_flow_edges:
                num_con_flow_edges += 1
            last_e = e    
        path_level_results.loc[q_id]['number of congested flow edges'] = num_con_flow_edges
        path_level_results.loc[q_id]['number of freeflow edges'] = len(path) - num_con_flow_edges
        path_level_results.loc[q_id]['penalty'] = round(visited[q[1]] - get_tt_of_path(q_best_path,eam),4)
        
    print()
    
cpu_end_time = time.time()

cpu_time = cpu_end_time - cpu_start_time

#%% Output System Results

system_level_results = pd.DataFrame(columns=['Algorithm','Sort','CPU Time', 'Total Travel Time','Count Quickest Paths','Count Non Quickest Paths','Total Congestion Penalty','Free Flow Edges Traversed','Congested Edges Traversed'])
zero_array = np.zeros(system_level_results.shape[1], dtype=int)

alg_name = 'Coll A-Star'
sort_by = exp_tag
alg_ind = len(system_level_results)
system_level_results.loc[alg_ind] = zero_array
system_level_results.xs(alg_ind)['Algorithm'] = alg_name
system_level_results.xs(alg_ind)['Sort'] = sort_by

system_level_results = output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp,alg_ind,alg_name,sort_by,base_folder)