## CSP Algorithms

#Algorithms and supporting functions for Colective Shortest Paths experiments

#Import Modules

from collections import defaultdict
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from csp_toolset import *

#%% ALGORITHM: Dijkstra

def dijsktra(G, source, target, edge_weight):
    visited = {source: 0}
    last_best = {}

    nodes = set(G.nodes)

    #Identify next node to land on
    while nodes: 
        current_node = None
        for node in nodes:
            if node in visited:
                if current_node is None:
                    current_node = node
                elif visited[node] < visited[current_node]:
                    current_node = node

        if current_node is None:
            break
        
        #when destination node reach target and get path
        if current_node == target:
            last_node = target
            path = [last_node]
            while last_node != source:
                last_node = last_best[last_node]
                path.append(last_node)
            break
        
        #Mark Visited Node
        nodes.remove(current_node)
        #Assign current length as up to current node
        current_weight = visited[current_node]

        for edge in G.edges(current_node,data=True):
            visited_node = edge[1]
            time_to_node =  edge[2][edge_weight]
            #Calculate time to next node
            weight = current_weight + time_to_node
            #If we have seen this node before, or if it represents a shorter route update visited matrix
            if visited_node not in visited or weight < visited[visited_node]:
                visited[visited_node] = weight
                last_best[visited_node] = current_node

    return visited[target], path[::-1]

#%% Capacity Aware Dijkstra

def capacity_aware_dijkstra(q,G,elm):
    
    #Assign source and target nodes
    source = q[0]
    target = q[1]
    
    #Set up matrix of nodes visited
    visited = {source: 0}
    #Set up list of best paths to node
    last_best = {}
    
    #Get all nodes
    nodes = set(G.nodes)
    
    last_node = None

    #For reporting    
    congested_flow_edges = []
    
    while nodes:
    
        #Identify next node to visit
        current_node = None
        for node in nodes:
            if node in visited:
                if current_node is None:
                    current_node = node
                elif visited[node] < visited[current_node]:
                    current_node = node
        
        if current_node is None:
            print('NOTE NOTE NOTE No Solution Found :' + str(q))
            return None,None,None
            break
            
        #Solution Found
        if current_node == target:
            last_node = target
            path = [last_node]
        
            while last_node != source:
                e_node = last_node
                last_node = last_best[last_node]
                path.append(last_node)
            
            return visited, path[::-1],congested_flow_edges
            break
        
        #Mark Visited Node
        nodes.remove(current_node)
        #Assign current length as up to current node
        current_weight = visited[current_node]
        
        #Get current time step
        t_at_u = q[2] + current_weight
        
        #Time interval lower and upper bounds
        t_lower = int(t_at_u)
        t_upper = math.ceil(t_at_u)
        
        for edge in G.edges(current_node,data=True):
            
            #Check whether free flow or congested flow
            free_flow_max = edge[2]['ff_max']            
            
            #Get Load on Edge at t
            try:
                #Get Load in Time Step
                edge_load_at_t = elm.xs((edge[0],edge[1]))[t_lower]
            except:
                edge_load_at_t = 0            
            
            #Get Delay Exponenet
            
            if edge_load_at_t <= free_flow_max:
                delay_exponent = 1
            else:
                delay_exponent = 1 / (edge_load_at_t - free_flow_max)
                congested_flow_edges.append([edge[0],edge[1]])
            
            #Arrival time at v
            t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge[2]['ff_travel_time'],4)    
            
            #Total Travel time of Path to V
            total_time_to_node = t_at_v - q[2]


            #Update tracking matrices if node not yet visisted, or time to node shorter than already recorded
            if (edge[1] not in visited) or (total_time_to_node < visited[edge[1]]):
                visited[edge[1]] = round(total_time_to_node,4)
                last_best[edge[1]] = current_node
        
        last_node = current_node

#%% Collective Top K

def collective_top_k(k_best_paths,q,eam,elm,t_best_path):

    penalties = []
    path_lengths = []
    path_vis_dictionaries = []
    congested_edge_count = []
    
    for i in range(1,len(k_best_paths.dropna()) + 1):
        path_to_check = eval(k_best_paths['path'+str(i)])
        
        path_vis = {}
        path_vis[q[0]] = 0
        congestested_edges = 0
        
        #Initiate
        t_start = q[2]
        current_tt = 0
        
        for i in range(0,len(path_to_check)-1):
            u_node = path_to_check[i]
            v_node = path_to_check[i+1]
            t_at_u = round(t_start + current_tt,4)
            edge_attributes = eam.xs((u_node,v_node))
            
            #Time interval lower and upper bounds
            t_lower = int(t_at_u)
            t_upper = math.ceil(t_at_u)
            
            #Get Load on Edge at t
            try:
                #Get Load in Time Step
                edge_load_at_t = elm.xs((u_node,v_node))[t_lower]
            except:
                edge_load_at_t = 0   
            
            #Get Delay Exponenet
            
            if edge_load_at_t <= edge_attributes['ff_max']:
                delay_exponent = 1
            else:
                delay_exponent = 1 / (edge_load_at_t - edge_attributes['ff_max'])
                congestested_edges += 1
                
            #Arrival time at v
            t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge_attributes['ff_travel_time'],4)                
            time_to_node = round(t_at_v - t_at_u,4)
            
            current_tt += time_to_node
            path_vis[v_node] = current_tt
        
        penalties.append(round(current_tt - t_best_path,4))
        path_lengths.append(round(current_tt,4))
        path_vis_dictionaries.append(path_vis)
        congested_edge_count.append(congestested_edges)
    
    path = eval(k_best_paths['path' + str(min(range(len(penalties)), key=penalties.__getitem__) + 1)])
    length = min(range(len(path_lengths)), key=path_lengths.__getitem__)
    visited = path_vis_dictionaries[min(range(len(penalties)), key=penalties.__getitem__)]
    num_con_flow_edges = congested_edge_count[min(range(len(penalties)), key=penalties.__getitem__)]
    
    return path, length, visited, num_con_flow_edges

#%% Load Aware A* Algorithm

def load_aware_a_star(q,G,elm,h_function,n2n_best_lengths):
    
    #Assign source and target nodes
    source = q[0]
    target = q[1]
    
    #Set up matrix of nodes visited
    
    #Total travel time up to nodes
    g_visited = {source: 0}
    f_visited = {source:h_function(q[0],q[1],n2n_best_lengths)}
    
    #Set up list of best paths to node
    last_best = {}
    
    #Get all nodes
    nodes = set(G.nodes)
    
    last_node = None
    
    #For reporting
    congested_flow_edges = []
    
    while nodes:
    
        #Identify next node to visit
        current_node = None
        current_Fscore = None
        for node in nodes:
            if node in f_visited:
                if current_node is None or f_visited[node] < current_Fscore:
                    current_Fscore  = f_visited[node]
                    current_node = node
        
        if current_node is None:
            print('NOTE NOTE NOTE No Solution Found :' + str(q))
            return None,None,None
            break
            
        #Solution Found
        if current_node == target:
            last_node = target
            path = [last_node]
        
            while last_node != source:
                e_node = last_node
                last_node = last_best[last_node]
                path.append(last_node)
            
            return g_visited, path[::-1],congested_flow_edges
            break
        
        #Mark Visited Node
        nodes.remove(current_node)
        #Assign current length as up to current node
        current_weight = g_visited[current_node]
        
        #Get current time step
        t_at_u = q[2] + current_weight
        
        #Time interval lower and upper bounds
        t_lower = int(t_at_u)
        t_upper = math.ceil(t_at_u)
        
        #For each edge coming out of node
        for edge in G.edges(current_node,data=True):
            
            #Check whether free flow or congested flow
            free_flow_max = edge[2]['ff_max']
            
            
            #Get Load on Edge at t
            try:
                #Get Load in Time Step
                edge_load_at_t = elm.xs((edge[0],edge[1]))[t_lower]
            except:
                edge_load_at_t = 0
            
            #Get Delay Exponenet
            
            if edge_load_at_t <= free_flow_max:
                delay_exponent = 1
            else:
                delay_exponent = 1 / (edge_load_at_t - free_flow_max)
                congested_flow_edges.append([edge[0],edge[1]])
            
            #Arrival time at v
            t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge[2]['ff_travel_time'],4)    
            
            #Total Travel time of Path to V
            total_time_to_node = t_at_v - q[2]
            
            #Update tracking matrices if node not yet visisted, or time to node shorter than already recorded
            if (edge[1] not in g_visited) or (total_time_to_node < g_visited[edge[1]]):
                g_visited[edge[1]] = total_time_to_node
                expected_time_to_source = best_path(edge[1],q[1],n2n_best_lengths)
                f_visited[edge[1]] = total_time_to_node + expected_time_to_source
                last_best[edge[1]] = current_node
        
        last_node = current_node

#%% A* Search with Load Prediction
   
def a_star_with_load_prediction(q,G,elm,h_function,n2n_best_lengths,exp_portion,actual_potion,model):

    #Set up blank feature vec
    u_nodes = pd.DataFrame(columns = list(G.nodes))
    v_nodes = pd.DataFrame(columns = list(G.nodes))

    #Assign source and target nodes
    source = q[0]
    target = q[1]
    
    #Set up matrix of nodes visited
    g_visited = {source: 0}
    f_visited = {source:h_function(q[0],q[1],n2n_best_lengths)}
    
    # Proportion of queries handled (%)
    actual_potion = 1 - exp_portion
    
    #Set up list of best paths to node
    last_best = {}
    
    #Get all nodes
    nodes = set(G.nodes)
    
    last_node = None
    
    while nodes:
        #Identify next node to visit
        current_node = None
        current_Fscore = None
        for node in nodes:
            if node in f_visited:
                if current_node is None or f_visited[node] < current_Fscore:
                    current_Fscore  = f_visited[node]
                    current_node = node
        
        if current_node is None:
            print('NOTE NOTE NOTE No Solution Found :' + str(q))
            return None,None
            break
            
        #Solution Found
        if current_node == target:
            last_node = target
            path = [last_node]
        
            while last_node != source:
                e_node = last_node
                last_node = last_best[last_node]
                path.append(last_node)

            return path[::-1]
            break
        
        #Mark Visited Node
        nodes.remove(current_node)
        #Assign current length as up to current node
        current_weight = g_visited[current_node]
        
        #Get current time step
        t_at_u = q[2] + current_weight
        
        #Time interval lower and upper bounds
        t_lower = int(t_at_u)
        t_upper = math.ceil(t_at_u)
        
        for edge in G.edges(current_node,data=True):
            
            #Check whether free flow or congested flow
            free_flow_max = edge[2]['ff_max']
            
            #Get Expected Load
            
            u_nodes.loc[0] = 0
            u_nodes.loc[0][edge[0]] = 1
            
            v_nodes.loc[0] = 0
            v_nodes.loc[0][edge[1]] = 1
            
            X = np.concatenate((u_nodes.values,v_nodes.values,np.array([[t_lower]])), axis=1).astype(int)
            
            expected_load = int(model.predict(X)[0][0])
            
            #Get Actual Load on Edge at t
            try:
                #Get Load in Time Step
                edge_load_at_t = elm.xs((edge[0],edge[1]))[t_lower]
            except:
                edge_load_at_t = 0
            
            predicted_load = int((exp_portion * expected_load) + (actual_potion * edge_load_at_t))
        
            #Get Delay Exponenet
            
            if predicted_load <= free_flow_max:
                delay_exponent = 1
            else:
                delay_exponent = 1 / (predicted_load - free_flow_max)
            
            #Arrival time at v
            t_at_v = round(t_lower + (t_at_u - t_lower)**delay_exponent + edge[2]['ff_travel_time'],4)    
            
            #Total Travel time of Path to V
            total_time_to_node = t_at_v - q[2]
            
            #Update tracking matrices if node not yet visisted, or time to node shorter than already recorded
            if (edge[1] not in g_visited) or (total_time_to_node < g_visited[edge[1]]):
                g_visited[edge[1]] = total_time_to_node
                expected_time_to_source = h_function(edge[1],q[1],n2n_best_lengths)
                f_visited[edge[1]] = total_time_to_node + expected_time_to_source
                last_best[edge[1]] = current_node
        last_node = current_node

#%% IPR - Search For Paths to Reassign

def ipr_iterate(stop_limit,cem,epm,elm,path_level_results,path_reassign_history,queries, eam,G,node_to_node_best_paths,path_to_congested_edges):
    #Set up Variables for reassign algorithm
    edge_check_hist = pd.DataFrame(columns=['edge','time'])
    full_edge_history = pd.DataFrame(columns=['edge','hold until'])
    lad_checks = 0
    avg_median_ratio = []
    stop_criteria_met = False
    
    i = -1
    
    while stop_criteria_met == False:
        
        i += 1
    
        # print('NEXT ITERATION : ' + str(i))
        
        hold_out_edges = full_edge_history.loc[full_edge_history['hold until'] >= i]['edge'].to_list()
        edge_found = False
        edge_index = -1
        
        #Sort congestion matrix
        sort_congested_edges = cem.stack()
        sort_congested_edges = sort_congested_edges.sort_values(ascending=False)
        
        
        while edge_found == False:
            edge_index += 1
            #Identify the most congested edge,t tuple in system    
            most_congested_edge = sort_congested_edges.index[edge_index]
            
            if most_congested_edge not in hold_out_edges:
                # print('Edge Found!')
                edge_found = True
        
        edge_check_hist = edge_check_hist.append({'edge': (most_congested_edge[0],most_congested_edge[1]), 'time': most_congested_edge[2]}, ignore_index=True)
        
        #Get all paths through this edge,t tuple - as candidate paths for reassignment
        cand_paths_for_reassign = epm.xs(tuple((most_congested_edge[0],most_congested_edge[1])))[most_congested_edge[2]]
        cand_paths_for_reassign = list(set(cand_paths_for_reassign))
        cand_paths_for_reassign = pd.DataFrame({'path id':cand_paths_for_reassign}).set_index('path id')
        cand_paths_for_reassign = pd.merge(path_level_results['penalty'], cand_paths_for_reassign, right_index=True, left_index=True, how='inner')
        cand_paths_for_reassign = cand_paths_for_reassign.sort_values(by=['penalty'], ascending=False)
        
        num_paths_to_check = len(cand_paths_for_reassign)
        
        j = -1
        progress = False
        
        while progress == False:
        
            j += 1
            
            # print('j : ' + str(j))
            
            #Get path ID (same as query ID)
            path_id_to_reassign = cand_paths_for_reassign.iloc[j].name
            # print('Checking ID : ' + str(path_id_to_reassign))
            #Get original query
            q = queries.loc[path_id_to_reassign]
            #Identify actual best path
            q_best_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
            q_best_path = eval(q_best_path)
            
            #Mark originally assigned path and length
            original_path = path_level_results.loc[path_id_to_reassign]['path']
            original_length = path_level_results.loc[path_id_to_reassign]['length']
            
            #Remove path from system
            
            #ELM
            elm = remove_path_to_elm(q,original_path,path_to_congested_edges[path_id_to_reassign][1],elm)
            
            #EPM    
            epm = remove_from_path_matrix(q,original_path,epm,path_id_to_reassign,path_to_congested_edges[path_id_to_reassign][1])
            
            #CEM
            cem = remove_path_cem(q,path_to_congested_edges[path_id_to_reassign][0],path_to_congested_edges[path_id_to_reassign][1],cem)
            
            #Reselect path using LAD
            visited,path,congested_flow_edges = capacity_aware_dijkstra(q,G,elm)
            lad_checks += 1
            
            path_t_saving = original_length - visited[q[1]]
            
            if path_t_saving > 0:
                progress = True
            else:
                # print('Time Savings : ' + str(path_t_saving))
                # print('No Benefit')
                #ELM
                elm = add_path_to_elm(q,original_path,path_to_congested_edges[path_id_to_reassign][1],elm)
                #EPM
                epm = update_path_matrix(q,original_path,epm,path_id_to_reassign,path_to_congested_edges[path_id_to_reassign][1])
                #CEM
                cem = add_path_cem(q,path_to_congested_edges[path_id_to_reassign][0],path_to_congested_edges[path_id_to_reassign][1],cem)        
            
            if j+1 == num_paths_to_check:
                # print('No Edges Left To Check...')
                full_edge_history = full_edge_history.append({'edge': most_congested_edge, 'hold until': i + 100}, ignore_index=True)
                break
        
        #Get Congested Edges for Path (for updating CEM)
        
        visited = {}
        visited[q[0]] = 0
        num_con_flow_edges = 0
        t_start = q[2]
        congested_edges = []
        
        total_time_to_node = 0
        
        for k in range(0,len(path)-1):
            u_node = path[k]
            v_node = path[k+1]
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
        
        new_path_length = round(total_time_to_node,4)
        
        #ELM
        elm = add_path_to_elm(q,path,visited,elm)
        #EPM
        epm = update_path_matrix(q,path,epm,path_id_to_reassign,visited)
        #CEM
        cem = add_path_cem(q,congested_edges,visited,cem)
        
        
        #Update Record Results
        path_level_results.loc[path_id_to_reassign]['path'] = path
        path_level_results.loc[path_id_to_reassign]['length'] = new_path_length
        path_level_results.loc[path_id_to_reassign]['best path'] = path == q_best_path
        path_level_results.loc[path_id_to_reassign]['best travel time'] = get_tt_of_path(q_best_path,eam)
        path_level_results.loc[path_id_to_reassign]['number of edges'] = len(path)
        path_level_results.loc[path_id_to_reassign]['number of congested flow edges'] = num_con_flow_edges
        path_level_results.loc[path_id_to_reassign]['number of freeflow edges'] = len(path) - num_con_flow_edges
        path_level_results.loc[path_id_to_reassign]['penalty'] = round(new_path_length - get_tt_of_path(q_best_path,eam),4)
        
        #Calcualte Savings Ratio
        #How much time saved per LAD call
        savings_ratio = path_t_saving / lad_checks
        
        #Update Path Re-Assign History
        path_reassign_history = path_reassign_history.append({'path_id': path_id_to_reassign, 'original path': original_path, 'new path': path, 'time saving': path_t_saving,'LAD Checks':lad_checks,'Savings Ratio':savings_ratio}, ignore_index=True)
        
        #Calcute Stopping Criteria    
        if path_reassign_history.shape[0] == 10:
            low_quart = path_reassign_history.tail(10)['Savings Ratio'].describe()['25%']
            median = path_reassign_history.tail(10)['Savings Ratio'].describe()['50%']
            upp_quart = path_reassign_history.tail(10)['Savings Ratio'].describe()['75%']
            
            savings_summary_stats = np.array([[low_quart,median,upp_quart]])
            
        elif path_reassign_history.shape[0] > 10:
            low_quart = path_reassign_history.tail(10)['Savings Ratio'].describe()['25%']
            median = path_reassign_history.tail(10)['Savings Ratio'].describe()['50%']
            upp_quart = path_reassign_history.tail(10)['Savings Ratio'].describe()['75%']
            
            savings_summary_stats = np.vstack([savings_summary_stats, [[low_quart,median,upp_quart]]])
        
            if savings_summary_stats.shape[0] >= 10:
                avg_median_ratio.append(savings_summary_stats[-10:].mean(axis = 0)[1])
                count_stop_criteria = 0
        
                for m in avg_median_ratio[-5:]:
                    # print('Average Median Saving Ratio : ' + str(m))
                    if m <= stop_limit:
                        count_stop_criteria += 1
                        
                if count_stop_criteria == 5:
                    stop_criteria_met = True
        
        lad_checks = 0
        path_to_congested_edges[path_id_to_reassign] = [congested_edges,visited]
        # print('Time Saving Ratio : ' + str(savings_ratio))
        # print('Time saved in this iteration : ' + str(path_t_saving))
        # print()
        
    return path_level_results,elm,epm,eam

#%% DATA TYPE: Edge Travel Time/Capacity Matrix

def edge_attribute_matrix(G):
    #Create edge load matrix
    edge_list = []
    for edge in G.edges(data=True):
        edge_append = [edge[0],edge[1],edge[2]['speed_limit'],edge[2]['length'],edge[2]['ff_max'],edge[2]['ff_travel_time']]
        edge_list.append(edge_append)
    attribute_matrix = pd.DataFrame(edge_list, columns=['source','target','speed_limit','length','ff_max','ff_travel_time'])
    attribute_matrix = attribute_matrix.set_index(['source','target']).sort_index()
    
    return attribute_matrix

def edge_attribute_dict(G):
    i = 0
    for edge in G.edges(data=True):
        if i == 0:
            attribute_dictionary = {tuple([edge[0],edge[1]]) : [edge[2]['speed_limit'],edge[2]['length'],edge[2]['ff_max'],edge[2]['ff_travel_time']]}
        else:
            attribute_dictionary[tuple([edge[0],edge[1]])] = [edge[2]['speed_limit'],edge[2]['length'],edge[2]['ff_max'],edge[2]['ff_travel_time']]
        i += 1
        
    return attribute_dictionary

#%% DATA TYPE - edge load matrix

def edge_load_matrix(G):
    #Create edge load matrix
    edge_list = []
    for edge in G.edges(data=True):
        edge_append = [edge[0],edge[1]]
        edge_list.append(edge_append)
    edge_cap_matrix = pd.DataFrame(edge_list, columns=['source','target'])
    edge_cap_matrix[1] = 0
    edge_cap_matrix = edge_cap_matrix.set_index(['source','target']).sort_index()
    return edge_cap_matrix

def edge_load_dictionary(G):
    eld = defaultdict(dict)
    for edge in G.edges(data=True):
        eld[tuple([edge[0],edge[1]])][1] = 0
    return eld

#%% DATA TYPE: Edge Path Matrix

def path_matrix(G):
    #Create edge load matrix
    edge_list = []
    for edge in G.edges(data=True):
        edge_append = [edge[0],edge[1]]
        edge_list.append(edge_append)
    #for edge in G.edges(data=True):
        #edge_append = [edge[1],edge[0]]
        #edge_list.append(edge_append)
    edge_cap_matrix = pd.DataFrame(edge_list, columns=['source','target'])
    edge_cap_matrix[1] = None
    edge_cap_matrix = edge_cap_matrix.set_index(['source','target']).sort_index()
    return edge_cap_matrix

def edge_path_dictionary(G):

    epd = defaultdict(dict)
    for edge in G.edges(data=True):
        epd[tuple([edge[0],edge[1]])][1] = []
    return epd

#%% Add a path to  ELM - requires weights to given edges in dictionary format

def add_path_to_elm(q,p,visited,elm):
    
    t = q[2]
    
    for i in range(1,len(p)):
        index_u = i-1
        index_v = i
        u_node = p[index_u]
        v_node = p[index_v]
        
        if i == 1:
            t_at_u = t
        else:
            t_at_u = t_at_v
        
        t_at_v  = t + visited[v_node]        
    
        if int(t_at_u) == int(t_at_v):
            try:
                elm.xs((u_node,v_node))[int(t_at_u)] += 1
            except:
                elm[int(t_at_u)] = 0
                elm.xs((u_node,v_node))[int(t_at_u)] += 1
        #Edge goes into next time interval
        else:
            for i in range(int(t_at_u),int(t_at_v)+1):
                try:
                    elm.xs((u_node,v_node))[i] += 1
                except:
                    elm[i] = 0
                    elm.xs((u_node,v_node))[i] += 1
        #Re-Index ELM
        elm = elm.reindex(sorted(elm.columns), axis=1)
        
    return elm

#%% Remove a path from ELM

def remove_path_to_elm(q,p,visited,elm):
    t = q[2]
    
    for i in range(1,len(p)):
        index_u = i-1
        index_v = i
        u_node = p[index_u]
        v_node = p[index_v]
        
        if i == 1:
            t_at_u = t
        else:
            t_at_u = t_at_v
        
        t_at_v  = t + visited[v_node]        
    
        if int(t_at_u) == int(t_at_v):
            elm.xs((u_node,v_node))[int(t_at_u)] -= 1
        #Edge goes into next time interval
        else:
            for i in range(int(t_at_u),int(t_at_v)+1):
                elm.xs((u_node,v_node))[i] -= 1
        #Re-Index ELM
        elm = elm.reindex(sorted(elm.columns), axis=1)
        
    return elm

#%% add new path to path matrix

def update_path_matrix(q,p,epm,q_id,visited):

    t = q[2]
    
    for i in range(1,len(p)):
        index_u = i-1
        index_v = i
        u_node = p[index_u]
        v_node = p[index_v]
        
        if i == 1:
            t_at_u = t
        else:
            t_at_u = t_at_v
        
        t_at_v  = t + visited[v_node] 

        #If edge starts and ends in same time interval
        if int(t_at_u) == int(t_at_v):
            try:
                if epm.xs((u_node,v_node))[int(t_at_u)] == None:
                    epm.xs((u_node,v_node))[int(t_at_u)] = [q_id]
                else:
                    epm.xs((u_node,v_node))[int(t_at_u)].append(q_id)
            except:
                epm[int(t_at_u)] = None
                epm.xs((u_node,v_node))[int(t_at_u)] = [q_id]
        else:
            for i in range(int(t_at_u),int(t_at_v)+1):
                try:
                    if epm.xs((u_node,v_node))[i] == None:
                        epm.xs((u_node,v_node))[i] = [q_id]
                    else:
                        epm.xs((u_node,v_node))[i].append(q_id)
                except:
                    epm[i] = None
                    epm.xs((u_node,v_node))[i] = [q_id]
                    
        epm = epm.reindex(sorted(epm.columns), axis=1)

    return epm

#%% Remove path from epm

def remove_from_path_matrix(q,p,epm,q_id,visited):

    t = q[2]
    
    for i in range(1,len(p)):
        index_u = i-1
        index_v = i
        u_node = p[index_u]
        v_node = p[index_v]
        
        if i == 1:
            t_at_u = t
        else:
            t_at_u = t_at_v
        
        t_at_v  = t + visited[v_node] 
    
        #If edge starts and ends in same time interval
        if int(t_at_u) == int(t_at_v):
            epm.xs((u_node,v_node))[int(t_at_u)].remove(q_id)
        else:
            for i in range(int(t_at_u),int(t_at_v)+1):
                epm.xs((u_node,v_node))[i].remove(q_id)                  
        epm = epm.reindex(sorted(epm.columns), axis=1)
        
    return epm


#%% Add new path CEM
    
def add_path_cem(q,congested_edges,visited,cem):
    t_start = q[2]
    
    for edge in congested_edges:
        t_at_u = visited[edge[0][0]] + t_start
        t_at_v = t_at_u + edge[1]
        
        if int(t_at_u) == int(t_at_v):
            try:
                cem.xs((edge[0][0],edge[0][1]))[int(t_at_u)] += 1
            except:
                cem[int(t_at_u)] = 0
                cem.xs((edge[0][0],edge[0][1]))[int(t_at_u)] += 1        
    
        #Edge goes into next time interval
        else:
            for i in range(int(t_at_u),int(t_at_v)+1):
                try:
                    cem.xs((edge[0][0],edge[0][1]))[i] += 1
                except:
                    cem[i] = 0
                    cem.xs((edge[0][0],edge[0][1]))[i] += 1
        #Re-Index ELM
        cem = cem.reindex(sorted(cem.columns), axis=1)
    
    return cem


#%% Remove path from CEM
    
def remove_path_cem(q,congested_edges,visited,cem):
    
    t_start = q[2]
    
    for edge in congested_edges:
        t_at_u = visited[edge[0][0]] + t_start
        t_at_v = t_at_u + edge[1]
        
        if int(t_at_u) == int(t_at_v):
            cem.xs((edge[0][0],edge[0][1]))[int(t_at_u)] -= 1      
    
        #Edge goes into next time interval
        else:
            for i in range(int(t_at_u),int(t_at_v)+1):
                cem.xs((edge[0][0],edge[0][1]))[i] -= 1

        #Re-Index ELM
        cem = cem.reindex(sorted(cem.columns), axis=1)
        
    return cem

#%% Load Aware Dijkstra Module
    
def alg_LAD(queries,node_to_node_best_paths,G,elm,epm,eam):

    path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])
    
    queries_handled = 0
    
    for i in range(0,len(queries)):
        q = queries.iloc[i].tolist()
        q_id = queries.iloc[i].name
        
        q_best_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
        q_best_path = eval(q_best_path)
        #print_path(G,q[0],q[1],q_best_path,'ff_travel_time')
        
        #Run Algorithm
        visited,path,congested_flow_edges = capacity_aware_dijkstra(q,G,elm)
        
        #Update Matrices
        elm = add_path_to_elm(q,path,visited,elm)
        epm = update_path_matrix(q,path,epm,q_id,visited)
        
        #Record Results
        path_level_results.loc[q_id]['path'] = path
        path_level_results.loc[q_id]['length'] = round(visited[q[1]],4)
        path_level_results.loc[q_id]['best path'] = path == q_best_path
        path_level_results.loc[q_id]['best travel time'] = round(get_tt_of_path(q_best_path,eam),4)
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
        
        queries_handled += 1
        # print(queries_handled)
    return path_level_results, elm, epm

#%% COllective Top K Module

def alg_collTopK(queries,node_to_node_best_paths,eam,elm,epm):
    
    path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])
    
    queries_handled = 0
    
    for i in range(0,len(queries)):
    
        q = queries.iloc[i].tolist()
        q_id = queries.iloc[i].name
        k_best_paths = node_to_node_best_paths.xs((q[0], q[1]))
        t_best_path = get_tt_of_path(eval(k_best_paths['path1']),eam)
        q_best_path = eval(k_best_paths['path1'])
        
        path, length, visited, num_con_flow_edges = collective_top_k(k_best_paths,q,eam,elm,t_best_path)
        
        elm = add_path_to_elm(q,path,visited,elm)
        epm = update_path_matrix(q,path,epm,q_id,visited)
        
        path_level_results.loc[q_id]['path'] = path
        path_level_results.loc[q_id]['length'] = visited[q[1]]
        path_level_results.loc[q_id]['best path'] = path == q_best_path
        path_level_results.loc[q_id]['best travel time'] = get_tt_of_path(q_best_path,eam)
        path_level_results.loc[q_id]['number of edges'] = len(path)
        path_level_results.loc[q_id]['number of congested flow edges'] = num_con_flow_edges
        path_level_results.loc[q_id]['number of freeflow edges'] = len(path) - num_con_flow_edges
        path_level_results.loc[q_id]['penalty'] = visited[q[1]] - get_tt_of_path(q_best_path,eam)
        
        queries_handled += 1
        # print(queries_handled)
        
    return path_level_results,elm,epm

#%% A* Module
    
def alg_AStar(queries,node_to_node_best_paths,G,elm,epm,eam,n2n_best_lengths):

    path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])
    
    queries_handled = 0
    
    for i in range(0,len(queries)):
        q = queries.iloc[i].tolist()
        q_id = queries.iloc[i].name
        
        q_best_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
        q_best_path = eval(q_best_path)
        #print_path(G,q[0],q[1],q_best_path,'ff_travel_time')
        
        #Run Algorithm
        visited,path,congested_flow_edges = load_aware_a_star(q,G,elm,best_path,n2n_best_lengths)
        
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
        path_level_results.loc[q_id]['penalty'] = visited[q[1]] - get_tt_of_path(q_best_path,eam)
        
        queries_handled += 1
        # print(queries_handled)
        
    return path_level_results,elm,epm

#%% A Star with Med Load Predictor
def alg_AStar_med_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,feature_vector,model,n2n_best_lengths):

    path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])
    
    for i in range(0,len(queries)):
        # print(str(i))
        q = queries.iloc[i].tolist()
        q_id = queries.iloc[i].name
        
        q_best_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
        q_best_path = eval(q_best_path)
        
        exp_portion = 1 - (i / len(queries))
        actual_potion = 1 - exp_portion
        
        path,figs,g_visited = a_star_with_load_prediction(q,G,elm,n2n_best_lengths,exp_portion,actual_potion,feature_vector,model)
        
        # Get actual path length
            
        visited = {}
        visited[q[0]] = 0
        num_con_flow_edges = 0
        congested_edges = []
        t_start = q[2]
        current_tt = 0
        
        for j in range(0,len(path)-1):
            s_node = path[j]
            e_node = path[j+1]
            t_now = t_start + current_tt
            edge_attributes = eam.xs((s_node,e_node))
            
            try:
                edge_load_at_t = elm.xs((s_node,e_node))[t_now]
            except:
                edge_load_at_t = 0
            
            residual_ff_capacity = edge_attributes['ff_max'] - edge_load_at_t
            
            if residual_ff_capacity > 0:
                #Get free flow time to next node
                time_to_node =  edge_attributes['ff_travel_time']
                #Calculate free flow time to next node
        
            #Edge Over-Capacity - Congestion flow
            else:
                #Get flow on node with additional vehicle
                congested_flow = edge_load_at_t + 1
                congested_speed = (edge_attributes['length'] / congested_flow) / 2
                time_to_node = int(round(edge_attributes['length'] / congested_speed))
                num_con_flow_edges += 1
                congested_edges.append([edge_attributes.name,time_to_node])
        
            current_tt += time_to_node
            visited[e_node] = current_tt
        
        actual_length = current_tt
        
        
        #Update system with new path
        
        #ELM
        elm = add_path_to_elm(q,path,visited,elm)
        #EPM
        epm = update_path_matrix(q,path,epm,q_id,visited)
        
        
        #Update Record Results
        path_level_results.loc[q_id]['path'] = path
        path_level_results.loc[q_id]['length'] = visited[q[1]]
        path_level_results.loc[q_id]['best path'] = path == q_best_path
        path_level_results.loc[q_id]['best travel time'] = get_tt_of_path(q_best_path,eam)
        path_level_results.loc[q_id]['number of edges'] = len(path)
        path_level_results.loc[q_id]['number of congested flow edges'] = num_con_flow_edges
        path_level_results.loc[q_id]['number of freeflow edges'] = len(path) - num_con_flow_edges
        path_level_results.loc[q_id]['penalty'] = visited[q[1]] - get_tt_of_path(q_best_path,eam)
        
    return path_level_results


#%% A Star with Weak Load Predictor
    
def alg_AStar_weak_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,feature_vector,model,n2n_best_lengths):

    path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])
    
    for i in range(0,len(queries)):
        # print(str(i))
        q = queries.iloc[i].tolist()
        q_id = queries.iloc[i].name
        
        q_best_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
        q_best_path = eval(q_best_path)
        
        exp_portion = (1 - (i / len(queries)))/2
        actual_potion = 1 - exp_portion
        
        path,figs,g_visited = a_star_with_load_prediction(q,G,elm,n2n_best_lengths,exp_portion,actual_potion,feature_vector,model)
        
        # Get actual path length
            
        visited = {}
        visited[q[0]] = 0
        num_con_flow_edges = 0
        congested_edges = []
        t_start = q[2]
        current_tt = 0
        
        for j in range(0,len(path)-1):
            s_node = path[j]
            e_node = path[j+1]
            t_now = t_start + current_tt
            edge_attributes = eam.xs((s_node,e_node))
            
            try:
                edge_load_at_t = elm.xs((s_node,e_node))[t_now]
            except:
                edge_load_at_t = 0
            
            residual_ff_capacity = edge_attributes['ff_max'] - edge_load_at_t
            
            if residual_ff_capacity > 0:
                #Get free flow time to next node
                time_to_node =  edge_attributes['ff_travel_time']
                #Calculate free flow time to next node
        
            #Edge Over-Capacity - Congestion flow
            else:
                #Get flow on node with additional vehicle
                congested_flow = edge_load_at_t + 1
                congested_speed = (edge_attributes['length'] / congested_flow) / 2
                time_to_node = int(round(edge_attributes['length'] / congested_speed))
                num_con_flow_edges += 1
                congested_edges.append([edge_attributes.name,time_to_node])
        
            current_tt += time_to_node
            visited[e_node] = current_tt
        
        actual_length = current_tt
        
        #
        
        #Update system with new path
        
        #ELM
        elm = add_path_to_elm(q,path,visited,elm)
        #EPM
        epm = update_path_matrix(q,path,epm,q_id,visited)
        
        
        #Update Record Results
        path_level_results.loc[q_id]['path'] = path
        path_level_results.loc[q_id]['length'] = visited[q[1]]
        path_level_results.loc[q_id]['best path'] = path == q_best_path
        path_level_results.loc[q_id]['best travel time'] = get_tt_of_path(q_best_path,eam)
        path_level_results.loc[q_id]['number of edges'] = len(path)
        path_level_results.loc[q_id]['number of congested flow edges'] = num_con_flow_edges
        path_level_results.loc[q_id]['number of freeflow edges'] = len(path) - num_con_flow_edges
        path_level_results.loc[q_id]['penalty'] = visited[q[1]] - get_tt_of_path(q_best_path,eam)
        
    return path_level_results

#%% A Star with Weak Strong Predictor
    
def alg_AStar_strong_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,feature_vector,model,n2n_best_lengths):

    path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])
    
    for i in range(0,len(queries)):
        # print(str(i))
        q = queries.iloc[i].tolist()
        q_id = queries.iloc[i].name
        
        q_best_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
        q_best_path = eval(q_best_path)
        
        actual_potion = (1 - (i / len(queries)))/2
        exp_portion = 1 - actual_potion
        
        path,figs,g_visited = a_star_with_load_prediction(q,G,elm,n2n_best_lengths,exp_portion,actual_potion,feature_vector,model)
        
        # Get actual path length
            
        visited = {}
        visited[q[0]] = 0
        num_con_flow_edges = 0
        congested_edges = []
        t_start = q[2]
        current_tt = 0
        
        for j in range(0,len(path)-1):
            s_node = path[j]
            e_node = path[j+1]
            t_now = t_start + current_tt
            edge_attributes = eam.xs((s_node,e_node))
            
            try:
                edge_load_at_t = elm.xs((s_node,e_node))[t_now]
            except:
                edge_load_at_t = 0
            
            residual_ff_capacity = edge_attributes['ff_max'] - edge_load_at_t
            
            if residual_ff_capacity > 0:
                #Get free flow time to next node
                time_to_node =  edge_attributes['ff_travel_time']
                #Calculate free flow time to next node
        
            #Edge Over-Capacity - Congestion flow
            else:
                #Get flow on node with additional vehicle
                congested_flow = edge_load_at_t + 1
                congested_speed = (edge_attributes['length'] / congested_flow) / 2
                time_to_node = int(round(edge_attributes['length'] / congested_speed))
                num_con_flow_edges += 1
                congested_edges.append([edge_attributes.name,time_to_node])
        
            current_tt += time_to_node
            visited[e_node] = current_tt
        
        actual_length = current_tt
        
        #
        
        #Update system with new path
        
        #ELM
        elm = add_path_to_elm(q,path,visited,elm)
        #EPM
        epm = update_path_matrix(q,path,epm,q_id,visited)
        
        
        #Update Record Results
        path_level_results.loc[q_id]['path'] = path
        path_level_results.loc[q_id]['length'] = visited[q[1]]
        path_level_results.loc[q_id]['best path'] = path == q_best_path
        path_level_results.loc[q_id]['best travel time'] = get_tt_of_path(q_best_path,eam)
        path_level_results.loc[q_id]['number of edges'] = len(path)
        path_level_results.loc[q_id]['number of congested flow edges'] = num_con_flow_edges
        path_level_results.loc[q_id]['number of freeflow edges'] = len(path) - num_con_flow_edges
        path_level_results.loc[q_id]['penalty'] = visited[q[1]] - get_tt_of_path(q_best_path,eam)
        
    return path_level_results

#%% A Star with Load Predictor
def alg_AStar_with_load_pred(queries,node_to_node_best_paths,G,elm,epm,eam,model,n2n_best_lengths,pred_strength):

    path_level_results = pd.DataFrame(index=queries.index, columns=['path','length','best path','best travel time','number of edges','number of freeflow edges','number of congested flow edges','penalty'])
    
    for i in range(0,len(queries)):
        
        # print(str(i))
        q = queries.iloc[i].tolist()
        q_id = queries.iloc[i].name
        
        q_best_path = node_to_node_best_paths.xs((q[0], q[1]))['path1']
        q_best_path = eval(q_best_path)
        
        # Proprtion of queries left (%)
        if pred_strength == 'Weak':
            exp_portion = (1 - (i / len(queries)))/2
            actual_potion = 1 - exp_portion
        elif pred_strength == 'Med':
            exp_portion = 1 - (i / len(queries))
            actual_potion = 1 - exp_portion
        else:
            actual_potion = (1 - (i / len(queries)))/2
            exp_portion = 1 - actual_potion
        
        #Get "best" path
        path = a_star_with_load_prediction(q,G,elm,best_path,n2n_best_lengths,exp_portion,actual_potion,model)
        
        # Get actual metrics for path (e.g. without predicted loads)
            
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
        epm = update_path_matrix(q,path,epm,q_id,visited)
        
        #Update Record Results
        path_level_results.loc[q_id]['path'] = path
        path_level_results.loc[q_id]['length'] = actual_length
        path_level_results.loc[q_id]['best path'] = path == q_best_path
        path_level_results.loc[q_id]['best travel time'] = get_tt_of_path(q_best_path,eam)
        path_level_results.loc[q_id]['number of edges'] = len(path)
        path_level_results.loc[q_id]['number of congested flow edges'] = num_con_flow_edges
        path_level_results.loc[q_id]['number of freeflow edges'] = len(path) - num_con_flow_edges
        path_level_results.loc[q_id]['penalty'] = round(actual_length - get_tt_of_path(q_best_path,eam),4)
        
    return path_level_results,elm,epm