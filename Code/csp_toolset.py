## CSP Toolset

#Set of tools e.g. visualisation, graph creation etc for collective shortest paths problem

import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np    
import math
# import scipy
# import pydot
from collections import defaultdict
from math import sqrt
import csv
import pickle
import os
# from ast import literal_eval
from itertools import islice

#%% Graph Set Up

def create_graph(m,n,c,t_dom):
    
    #Define dimensions of graphs depending on input parameters
    #Only allow 1, 4, 9 or 16 clusters
    #Only allow edges lengths (e.g. m and n) to be equally divisible by value of c
    #Clusters created using new calculated parameters m_c and n_c which give the dimension of each cluster - clusters will always be of exactly equal size
    #Use MultiDiGraph- a directed graph (representing direction of travel on a road) and multuple edges between nodes (allowing two way streets)
    
    if c == 1:
        m_c = m
        n_c = n
        G = nx.DiGraph()
        
    elif c == 4:
        m_c = m / 2
        n_c = n / 2
        
        if float(m_c).is_integer() and float(n_c).is_integer():
            m_c = int(m_c)
            n_c = int(n_c)
            G = nx.DiGraph()
        else:
            print('Cannot divide into equal clusters')
            print('Please select an alternative m and n')
            return
            
    elif c == 9:
        m_c = m / 3
        n_c = n / 3
        
        if float(m_c).is_integer() and float(n_c).is_integer():
            m_c = int(m_c)
            n_c = int(n_c)
            G = nx.DiGraph()
        else:
            print('Cannot divide into equal clusters')
            print('Please select an alternative m and n')
            return
            
    elif c == 16:
        m_c = m / 4
        n_c = n / 4
        
        if float(m_c).is_integer() and float(n_c).is_integer():
            m_c = int(m_c)
            n_c = int(n_c)
            G = nx.DiGraph()
        else:
            print('Cannot divide into equal clusters')
            print('Please select an alternative m and n')
            return
            
    else:
        print('Please select either 1, 4, 9 or 16 clusters')
        return
    
    #Add Nodes
    #Node naming in the format (c,m,n) as a tuple
    nodes = []
    
    for _c in range(1,c+1):
        for _m in range(1,m_c+1):
            for _n in range(1,n_c+1):
                node_id = (_c,_m,_n)
                nodes.append(node_id)
    G.add_nodes_from(nodes)

    #Add edges "going down"
    #Add randomised parameters travel time and capacity
    #Add edge going in either direction
    for _c in range(1,c+1):
        for _m in range(1,m_c+1):
            #Select random numer which will determine all values long ways across grid
            edge_lengths, edge_speeds = get_edge_attributes(n_c)
            for _n in range(1,n_c):
                #Add Edge Length
                edge_length = edge_lengths[_n-1]                
                
                #Add Edge Speed Limit
                speed_limit = edge_speeds[_n-1]
                
                #Headway
                if speed_limit > 20:
                    headway = 5
                else:
                    headway = 4

                #Calculate free flow travel time - in quarter hour domain
                ff_travel_time = round(((edge_length / speed_limit) / t_dom) ,4)
                
                #Calculate Free Flow Max
                ff_max = math.ceil((edge_length / (speed_limit * headway)) + (t_dom/(headway+((ff_travel_time * 360)/2))))
                
                #Add Attributes
                start_node = (_c,_m,_n)
                end_node = (_c,_m,_n+1)
                G.add_edge(start_node,end_node,speed_limit = speed_limit,length = edge_length,ff_max = ff_max,ff_travel_time = ff_travel_time)
                G.add_edge(end_node,start_node,speed_limit = speed_limit,length = edge_length,ff_max = ff_max,ff_travel_time = ff_travel_time)
    
    for _c in range(1,c+1):
        for _n in range(1,n_c+1):
            #Select random numer which will determine all values long ways across grid
            edge_lengths, edge_speeds = get_edge_attributes(m_c)
            for _m in range(1,m_c):
                #Add Edge Length
                edge_length = edge_lengths[_m-1]                
                
                #Add Edge Speed Limit
                speed_limit = edge_speeds[_m-1]
                
                #Headway
                if speed_limit > 20:
                    headway = 5
                else:
                    headway = 4
                
                #Calculate Free Flow Max
                ff_max = math.ceil((edge_length / (speed_limit * headway)) + (t_dom/(headway+((ff_travel_time * 360)/2))))
                
                #Calculate free flow travel time
                ff_travel_time = round(((edge_length / speed_limit) / t_dom) ,4)
                
                #Add Attributes
                start_node = (_c,_m,_n)
                end_node = (_c,_m+1,_n)
                G.add_edge(start_node,end_node,speed_limit = speed_limit,length = edge_length,ff_max = ff_max,ff_travel_time = ff_travel_time)
                G.add_edge(end_node,start_node,speed_limit = speed_limit,length = edge_length,ff_max = ff_max,ff_travel_time = ff_travel_time)
    
    return G,m_c,n_c

def create_cluster_mapping(c):

    cluster_map = defaultdict(dict)
    
    if c == 1:
        cluster_map[1]['u'] = None
        cluster_map[1]['r'] = None
        cluster_map[1]['d'] = None
        cluster_map[1]['l'] = None
        clust_to_row = {1:1}
        clust_to_col = {1:1}
    
    elif c == 4:
        cluster_map[1]['u'] = None
        cluster_map[1]['r'] = 2
        cluster_map[1]['d'] = 3
        cluster_map[1]['l'] = None
        
        cluster_map[2]['u'] = None
        cluster_map[2]['r'] = None
        cluster_map[2]['d'] = 4
        cluster_map[2]['l'] = 1
        
        cluster_map[3]['u'] = 1
        cluster_map[3]['r'] = 4
        cluster_map[3]['d'] = None
        cluster_map[3]['l'] = None
        
        cluster_map[4]['u'] = 2
        cluster_map[4]['r'] = None
        cluster_map[4]['d'] = None
        cluster_map[4]['l'] = 3   
        
        clust_to_row = {1:1,2:1,3:2,4:2}
        clust_to_col = {1:1,2:2,3:1,4:2}
    
    elif c == 9:
        cluster_map[1]['u'] = None
        cluster_map[1]['r'] = 2
        cluster_map[1]['d'] = 4
        cluster_map[1]['l'] = None
        
        cluster_map[2]['u'] = None
        cluster_map[2]['r'] = 3
        cluster_map[2]['d'] = 5
        cluster_map[2]['l'] = 1
        
        cluster_map[3]['u'] = None
        cluster_map[3]['r'] = None
        cluster_map[3]['d'] = 6
        cluster_map[3]['l'] = 2
        
        cluster_map[4]['u'] = 1
        cluster_map[4]['r'] = 5
        cluster_map[4]['d'] = 7
        cluster_map[4]['l'] = None
        
        cluster_map[5]['u'] = 2
        cluster_map[5]['r'] = 6
        cluster_map[5]['d'] = 8
        cluster_map[5]['l'] = 4
        
        cluster_map[6]['u'] = 3
        cluster_map[6]['r'] = None
        cluster_map[6]['d'] = 9
        cluster_map[6]['l'] = 5
        
        cluster_map[7]['u'] = 4
        cluster_map[7]['r'] = 8
        cluster_map[7]['d'] = None
        cluster_map[7]['l'] = None
        
        cluster_map[8]['u'] = 5
        cluster_map[8]['r'] = 9
        cluster_map[8]['d'] = None
        cluster_map[8]['l'] = 7
        
        cluster_map[9]['u'] = 6
        cluster_map[9]['r'] = None
        cluster_map[9]['d'] = None
        cluster_map[9]['l'] = 8
        
        clust_to_row = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3}
        clust_to_col = {1:1,2:2,3:3,4:1,5:2,6:3,7:1,8:2,9:3}
        
    elif c == 16:
        cluster_map[1]['u'] = None
        cluster_map[1]['r'] = 2
        cluster_map[1]['d'] = 5
        cluster_map[1]['l'] = None
        
        cluster_map[2]['u'] = None
        cluster_map[2]['r'] = 3
        cluster_map[2]['d'] = 6
        cluster_map[2]['l'] = 1
        
        cluster_map[3]['u'] = None
        cluster_map[3]['r'] = 4
        cluster_map[3]['d'] = 7
        cluster_map[3]['l'] = 2
        
        cluster_map[4]['u'] = None
        cluster_map[4]['r'] = None
        cluster_map[4]['d'] = 8
        cluster_map[4]['l'] = 3
        
        cluster_map[5]['u'] = 1
        cluster_map[5]['r'] = 6
        cluster_map[5]['d'] = 9
        cluster_map[5]['l'] = None
        
        cluster_map[6]['u'] = 2
        cluster_map[6]['r'] = 7
        cluster_map[6]['d'] = 10
        cluster_map[6]['l'] = 5
        
        cluster_map[7]['u'] = 3
        cluster_map[7]['r'] = 8
        cluster_map[7]['d'] = 11
        cluster_map[7]['l'] = 6
        
        cluster_map[8]['u'] = 4
        cluster_map[8]['r'] = None
        cluster_map[8]['d'] = 12
        cluster_map[8]['l'] = 7
        
        cluster_map[9]['u'] = 5
        cluster_map[9]['r'] = 10
        cluster_map[9]['d'] = 13
        cluster_map[9]['l'] = None        
        
        cluster_map[10]['u'] = 6
        cluster_map[10]['r'] = 11
        cluster_map[10]['d'] = 14
        cluster_map[10]['l'] = 9
        
        cluster_map[11]['u'] = 7
        cluster_map[11]['r'] = 12
        cluster_map[11]['d'] = 15
        cluster_map[11]['l'] = 10
        
        cluster_map[12]['u'] = 8
        cluster_map[12]['r'] = None
        cluster_map[12]['d'] = 16
        cluster_map[12]['l'] = 11
        
        cluster_map[13]['u'] = 9
        cluster_map[13]['r'] = 14
        cluster_map[13]['d'] = None
        cluster_map[13]['l'] = None
        
        cluster_map[14]['u'] = 10
        cluster_map[14]['r'] = 15
        cluster_map[14]['d'] = None
        cluster_map[14]['l'] = 13
        
        cluster_map[15]['u'] = 11
        cluster_map[15]['r'] = 16
        cluster_map[15]['d'] = None
        cluster_map[15]['l'] = 14
        
        cluster_map[16]['u'] = 12
        cluster_map[16]['r'] = None
        cluster_map[16]['d'] = None
        cluster_map[16]['l'] = 15
        
        clust_to_row = {1:1,2:1,3:1,4:1,5:2,6:2,7:2,8:2,9:3,10:3,11:3,12:3,13:4,14:4,15:4,16:4}
        clust_to_col = {1:1,2:2,3:3,4:4,5:1,6:2,7:3,8:4,9:1,10:2,11:3,12:4,13:1,14:2,15:3,16:4}
    
    return cluster_map,clust_to_row,clust_to_col

def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items() if length <= n]


def get_edge_attributes(num_edges):

    lengths = []    
    speeds = []
    road_in_range = False
    
    while road_in_range == False:
    
        initial_rand_length = np.random.normal(400,50,1)
        speed_limit_seed = rand_numer = random.uniform(0, 1)
        if initial_rand_length >= 250 and initial_rand_length <= 550:
            road_in_range = True
    
    for i in range(0,num_edges):
        
        #Get Edge Length
        actual_road_length = np.random.normal(initial_rand_length,5,1)
        
        #Get Speed Limit
        y_speed_limit_noise = np.random.normal(rand_numer,0.05,1)[0]
        if y_speed_limit_noise <= 0:
            y_speed_clip = 0.001
        elif y_speed_limit_noise >= 1:
            y_speed_clip = 0.999
        else:
            y_speed_clip = y_speed_limit_noise
        y_speed = -math.log(1 - (1 - math.exp(-3)) * y_speed_clip) / 3  
        speed_limit = (int(round(0.5 + (3 * y_speed)))) * 10    
        
        lengths.append(actual_road_length[0])
        speeds.append(speed_limit)
    
    return lengths, speeds

#%% Visualisation Tools

def map_colours(G,all_cbds,all_suburbs):
    color_map = []
    
    for node in G:
        if node in all_cbds:
            color_map.append('forestgreen')
        elif node in all_suburbs:
            color_map.append('dodgerblue')
        else:
            color_map.append('slategray')
    
    return color_map

def show_graph_all(G,all_cbds,all_suburbs):
    try:
        color_map = map_colours(G,all_cbds,all_suburbs)
    except:
        color_map = 'slategray'
    pos = nx.nx_pydot.pydot_layout(G)
    nx.draw(G,pos,node_color = color_map,with_labels = True)
    plt.show()
    
    
def show_graph_bridges(G,all_cbds,all_suburbs):
    try:
        color_map = map_colours(G,all_cbds,all_suburbs)
    except:
        color_map = 'slategray'
    nx.draw_spectral(G,node_color = color_map)
    plt.show()
    
def show_graph_clusters(G,c,nodes_for_bridges,all_cbds,all_suburbs):
    if c == 4:
        fig, axes = plt.subplots(nrows=2, ncols=2)
        ax = axes.flatten()
        for i in range(4):
            nodes_to_display = [tuple(l) for l in nodes_for_bridges[nodes_for_bridges['c'] == (i+1)].values.tolist()]
            H = G.subgraph(nodes_to_display)
            try:
                color_map = map_colours(H,all_cbds,all_suburbs)
            except:
                color_map = 'slategray'
            pos = nx.nx_pydot.pydot_layout(H)
            nx.draw(H,pos,node_color = color_map,with_labels = True, ax=ax[i])
            ax[i].set_axis_off()
        
        plt.show()    
    
    elif c == 9:
        fig, axes = plt.subplots(nrows=3, ncols=3)
        ax = axes.flatten()
        for i in range(9):
            nodes_to_display = [tuple(l) for l in nodes_for_bridges[nodes_for_bridges['c'] == (i+1)].values.tolist()]
            H = G.subgraph(nodes_to_display)
            try:
                color_map = map_colours(H,all_cbds,all_suburbs)
            except:
                color_map = 'slategray'
            pos = nx.nx_pydot.pydot_layout(H)
            nx.draw(H,pos,node_color = color_map,with_labels = True, ax=ax[i])
            ax[i].set_axis_off()
        
        plt.show()
        
    elif c == 16:
        fig, axes = plt.subplots(nrows=4, ncols=4)
        ax = axes.flatten()
        for i in range(16):
            nodes_to_display = [tuple(l) for l in nodes_for_bridges[nodes_for_bridges['c'] == (i+1)].values.tolist()]
            H = G.subgraph(nodes_to_display)
            try:
                color_map = map_colours(H,all_cbds,all_suburbs)
            except:
                color_map = 'slategray'
            pos = nx.nx_pydot.pydot_layout(H)
            nx.draw(H,pos,node_color = color_map,with_labels = True, ax=ax[i])
            ax[i].set_axis_off()
        
        plt.show()
        
def show_subset(G,focal_node,span,cluster_map,n_c,m_c,all_cbds,all_suburbs):
    
    c_r = focal_node[0]
    m_r = focal_node[1]
    n_r = focal_node[2]
    
    row_up = 0
    
    for i in range(span):
        if n_r == 1 and cluster_map[c_r]['u'] != None:
            n_r = n_c
            c_r = cluster_map[c_r]['u']
            row_up += 1
        elif n_r == 1 and cluster_map[c_r]['u'] == None:
            break
        else:
            n_r -= 1
            row_up += 1
    
    top_row = [tuple([c_r,m_r,n_r])]
    all_nodes_display = [tuple([c_r,m_r,n_r])]
    
    _m_r = m_r
    _c_r = c_r
    
    for i in range(span):
        if m_r == 1 and cluster_map[c_r]['l'] != None:
            m_r = m_c
            c_r = cluster_map[c_r]['l']
        elif m_r == 1 and cluster_map[c_r]['l'] == None:
            break
        else:
            m_r -= 1
        
        top_row.append(tuple([c_r,m_r,n_r]))
        all_nodes_display.append(tuple([c_r,m_r,n_r]))
       
    m_r = _m_r
    c_r = _c_r
    
    for i in range(span):
        if m_r == m_c and cluster_map[c_r]['r'] != None:
            m_r = 1
            c_r = cluster_map[c_r]['r']
        elif m_r == m_c and cluster_map[c_r]['r'] == None:
            break
        else:
            m_r += 1
        
        top_row.append(tuple([c_r,m_r,n_r]))
        all_nodes_display.append(tuple([c_r,m_r,n_r]))
    
    for n in top_row:
        c_r = n[0]
        m_r = n[1]
        n_r = n[2]
        
        for i in range(span * 2 - (span - row_up)):
            if n_r == n_c and cluster_map[c_r]['d'] != None:
                n_r = 1
                c_r = cluster_map[c_r]['d']
            elif n_r == n_c and cluster_map[c_r]['d'] == None:
                break
            else:
                n_r += 1
        
            all_nodes_display.append(tuple([c_r,m_r,n_r]))
    
    H = G.subgraph(all_nodes_display)
    pos = nx.nx_pydot.pydot_layout(H)
    try:
        color_map = map_colours(H,all_cbds,all_suburbs)
    except:
        color_map = 'slategray'
    nx.draw(H,pos,node_color = color_map,with_labels = True)
    plt.show()

def print_path(G,source,target,path,edge_weight):
    node_color_map = []
    for node in G:
        if node == source:
            node_color_map.append('green')
        elif node == target:
            node_color_map.append('red')
        else:
            node_color_map.append('slategray')
    
    edge_colour_map = []
    edge_weights = []
    
    for edge in G.edges:
        if edge[0] in path and edge[1] in path:
            edge_colour_map.append('lightseagreen')
            edge_weights.append(5)
        else:
            edge_colour_map.append('slategray')
            edge_weights.append(3)
            
    fig, ax = plt.subplots(figsize=(10,10))
    pos = nx.nx_pydot.pydot_layout(G)
    nx.draw(G, pos, node_color = node_color_map, edge_color=edge_colour_map, with_labels = True, width=edge_weights)
    edge_labels = nx.get_edge_attributes(G,edge_weight)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    plt.show()
    return fig

def print_dijkstra_move(G, current_node, matrices_updated, visited_nodes_for_print, source, target,congested_edge_print,edge_weight):
    
    node_color_map = []
    for node in G:
        if node == current_node:
            node_color_map.append('yellow')
        elif node in matrices_updated:
            node_color_map.append('lightgreen')
        # elif node in visited_nodes_for_print:
        #     node_color_map.append('orange')
        elif node == source:
            node_color_map.append('green')
        elif node == target:
            node_color_map.append('red')
        else:
            node_color_map.append('slategray')
    
    edge_colour_map = []
    edge_weights = []
    
    for edge in G.edges:
        if edge[0] == current_node and edge[1] in congested_edge_print:
            edge_colour_map.append('red')
            edge_weights.append(5)
        elif edge[0] == current_node and edge[1] in visited_nodes_for_print:
            edge_colour_map.append('lightgreen')
            edge_weights.append(5)
        else:
            edge_colour_map.append('slategray')
            edge_weights.append(0.3)
            
    fig, ax = plt.subplots(figsize=(10,10))
    pos = nx.nx_pydot.pydot_layout(G)
    nx.draw(G, pos, node_color = node_color_map, edge_color=edge_colour_map,with_labels = True,width=edge_weights)
    edge_labels = nx.get_edge_attributes(G,edge_weight)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    plt.show()
    
    return fig


#%% Add Bridges

#Get number of brdiges to add
#Given as random interger value between 1/3 and 1/2 of input dimension - will always return at least 1

def get_number_bridges(dim,low_lim,upp_lim):
    lower_bound = math.ceil(dim * low_lim)
    upper_bound = int(dim * upp_lim)
    num_bridges = random.randint(lower_bound, upper_bound)
    
    return num_bridges

#Add bridge to graph

def add_bridge(c1,c2,b,G,t_dom):
    start_node = tuple(c1.iloc[b].to_list())
    end_node = tuple(c2.iloc[b].to_list())
    speed_limit =  get_edge_attributes(1)[1][0]
    edge_length = get_edge_attributes(1)[0][0]
    
    #Headway
    if speed_limit > 20:
        headway = 5
    else:
        headway = 4
        
    ff_travel_time = round(((edge_length / speed_limit) / t_dom) ,4)
    #max capacity at free flow e.g. how many vehicle can travel at speed limit while maintaining headway
    ff_max = math.ceil((edge_length / (speed_limit * headway)) + (t_dom/(headway+((ff_travel_time * 360)/2))))
    

    #Add Edges
    G.add_edge(start_node,end_node,speed_limit = speed_limit,length = edge_length,ff_max = ff_max,ff_travel_time = ff_travel_time)
    G.add_edge(end_node,start_node,speed_limit = speed_limit,length = edge_length,ff_max = ff_max,ff_travel_time = ff_travel_time)
    return G

#Identify nodes which can be used to link to other clusters
#These will always be on one or other edge of a cluster
#Nodes along edges can be identified with following formulas:
#Rigt edge: c,m_c,rand(1,n_c)
#Left edge: c,1,rand(1,n_c)
#Lower edge: c,rand(1,m_c),n_c
#Upper edge: c,rand(1,m_c),1

#Select nodes as per above formula for each cluster
#Put nodes to dataframe and randomly sort dataframe
#When selecting nodes to link - take from top of dataframes to ensure that nodes are randomly selected and that no node can be selected twice

def add_bridges(G,c,m_c,n_c,low_lim,upp_lim,t_dom):

    nodes_for_bridges = pd.DataFrame.from_records(list(G.nodes()))
    nodes_for_bridges.columns = ['c','m','n']
    
    for _c in range(1,c+1):
        #Discover right edge nodes
        globals()['r_'+str(_c)] = nodes_for_bridges[(nodes_for_bridges['c'] == _c) & (nodes_for_bridges['m'] == m_c)]
        globals()['r_'+str(_c)] = globals()['r_'+str(_c)].sample(frac=1)
        #Discover down edge nodes
        globals()['d_'+str(_c)] = nodes_for_bridges[(nodes_for_bridges['c'] == _c) & (nodes_for_bridges['n'] == n_c)]
        globals()['d_'+str(_c)] = globals()['d_'+str(_c)].sample(frac=1)
        #Discover left edge nodes
        globals()['l_'+str(_c)] = nodes_for_bridges[(nodes_for_bridges['c'] == _c) & (nodes_for_bridges['m'] == 1)]
        globals()['l_'+str(_c)] = globals()['l_'+str(_c)].sample(frac=1)
        #Discover upper edge nodes
        globals()['u_'+str(_c)] = nodes_for_bridges[(nodes_for_bridges['c'] == _c) & (nodes_for_bridges['n'] == 1)]
        globals()['u_'+str(_c)] = globals()['u_'+str(_c)].sample(frac=1)
    
    # build bridges
    
    num_bridges_across = get_number_bridges(n_c,low_lim,upp_lim)
    num_bridges_down = get_number_bridges(m_c,low_lim,upp_lim)
    
    if c == 4:
        #Link r_1 to l_2
        
        for b in range(1,num_bridges_across+1):
                G = add_bridge(r_1,l_2,b,G,t_dom)
        
        #Link d_1 to u_3
        for b in range(1,num_bridges_down+1):
                G = add_bridge(d_1,u_3,b,G,t_dom)
            
        #Link d_2 to u_4
        for b in range(1,num_bridges_down+1):
                G = add_bridge(d_2,u_4,b,G,t_dom)    
        
        #link r_3 to l_4
        for b in range(1,num_bridges_across+1):
                G = add_bridge(r_3,l_4,b,G,t_dom)
                
    elif c == 9:
        
        #r_1 to l_2
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_1,l_2,b,G,t_dom)
        #r_2 to l_3
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_2,l_3,b,G,t_dom)
        #r_4 to l_5
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_4,l_5,b,G,t_dom)
        #r_5 to l_6
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_5,l_6,b,G,t_dom)
        #r_7 to l_8
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_7,l_8,b,G,t_dom)
        #r_8 to l_9
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_8,l_9,b,G,t_dom)
        #d_1 to u_4
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_1,u_4,b,G,t_dom)
        #d_4 to u_7
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_4,u_7,b,G,t_dom)
        #d_2 to u_5
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_2,u_5,b,G,t_dom)
        #d_5 to u_8
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_5,u_8,b,G,t_dom)
        #d_3 to u_6
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_3,u_6,b,G,t_dom)
        #d_6 to u_9
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_6,u_9,b,G,t_dom)
    
    
    elif c == 16:
        
        #r_1 to l_2
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_1,l_2,b,G,t_dom)
        #r_2 to l_3
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_2,l_3,b,G,t_dom)
        #r_3 to l_4
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_3,l_4,b,G,t_dom)
        #r_5 to l_6
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_5,l_6,b,G,t_dom)
        #r_6 to l_7
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_6,l_7,b,G,t_dom)
        #r_7 to l_8
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_7,l_8,b,G,t_dom)
        #r_9 to l_10
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_9,l_10,b,G,t_dom)
        #r_10 to l_11
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_10,l_11,b,G,t_dom)
        #r_11 to l_12
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_11,l_12,b,G,t_dom)
        #r_13 to l_14
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_13,l_14,b,G,t_dom)
        #r_14 to l_15
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_14,l_15,b,G,t_dom)
        #r_15 to l_16
        for b in range(1,num_bridges_across+1):
            G = add_bridge(r_15,l_16,b,G,t_dom)
        #d_1 to u_5
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_1,u_5,b,G,t_dom)
        #d_5 to u_9
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_5,u_9,b,G,t_dom)
        #d_9 to u_13
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_9,u_13,b,G,t_dom)
        #d_2 to u_6
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_2,u_6,b,G,t_dom)
        #d_6 to u_10
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_6,u_10,b,G,t_dom)
        #d_10 to u_14
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_10,u_14,b,G,t_dom)
        #d_3 to u_7
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_3,u_7,b,G,t_dom)
        #d_7 to u_11
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_7,u_11,b,G,t_dom)
        #d_11 to u_15
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_11,u_15,b,G,t_dom)
        #d_4 to u_8
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_4,u_8,b,G,t_dom)
        #d_8 to u_12
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_8,u_12,b,G,t_dom)
        #d_12 to u_16
        for b in range(1,num_bridges_down+1):
            G = add_bridge(d_12,u_16,b,G,t_dom)
    
    return G, nodes_for_bridges

#%% Define CBDs and Suburbs

#Methods:
#    Iterate through clusters
#    For each clusters randomise list of nodes and select into sub graph
#    Take first node in list as cbd
#    Then select suburb 1 and 2 ensuring they are at least two nodes away
#    The grow clusters
#        Expands by a range on 1 node on each iteration
#        Once a node is designated to a cbd, suburb etc it is fixed
#        Initial priority given to CBG e.g. grow the cluster first then then the suburbs
#        Stop after certain coverage reached

def cbds_suburbs(G,c,nodes_for_bridges,all_cbds,all_suburbs,cov_parm):
    for clust in range(1,c+1):
    
        #Select seed nodes for CBD and Suburbs
        
        cluster_nodes = nodes_for_bridges[(nodes_for_bridges['c'] == clust)]
        H = G.subgraph([tuple(l) for l in cluster_nodes.values.tolist()])
        cluster_nodes = cluster_nodes.sample(frac=1)
        
        #Track Nodes Used
        cbd = [tuple(cluster_nodes.iloc[0].tolist())]
        all_cbds.append(cbd[0])
        
        next_node = 1
        node_selected = False
        
        try:
            while node_selected == False:
                node_to_test = tuple(cluster_nodes.iloc[next_node].tolist())
                if node_to_test not in neighborhood(H,cbd[0],2):
                    sub_1 = [node_to_test]
                    all_suburbs.append(sub_1[0])
                    node_selected = True
                next_node += 1
            
            node_selected = False
        except:
            print('ERROR - Could not place CBD and Suburbs')
            print('Try again, or expand size of grid')
            break
        
        try:
            while node_selected == False:
                node_to_test = tuple(cluster_nodes.iloc[next_node].tolist())
                
                if node_to_test not in neighborhood(H,cbd[0],2) and node_to_test not in neighborhood(H,sub_1[0],2):
                    sub_2 = [node_to_test]
                    all_suburbs.append(sub_2[0])
                    node_selected = True
                next_node += 1
        except:
            print('ERROR - Could not place CBD and Suburbs')
            print('Try again, or expand size of grid')
            break
        nodes_spoken_for = []
        
        # Grow Clusters
        coverage = 0
        i = 0
        try:
            while coverage <= cov_parm:
            
                cbd = neighborhood(H,cbd[0],i)
                cbd = [n for n in cbd if n not in nodes_spoken_for]
                nodes_spoken_for = nodes_spoken_for + cbd
                
                sub_1 = neighborhood(H,sub_1[0],i)
                sub_1 = [n for n in sub_1 if n not in nodes_spoken_for]
                nodes_spoken_for = nodes_spoken_for + sub_1
                
                sub_2 = neighborhood(H,sub_2[0],i)
                sub_2 = [n for n in sub_2 if n not in nodes_spoken_for]
                nodes_spoken_for = nodes_spoken_for + sub_2
                
                all_cbds = list(set(all_cbds + cbd))
                all_suburbs = list(set(all_suburbs + sub_1 + sub_2))
                
                coverage = float(float(len(nodes_spoken_for)) / float(len(list(H.nodes()))))        
                i += 1
        except:
            print('ERROR - Could not place CBD and Suburbs')
            print('Try again, or expand size of grid')
            break
    
    all_other_nodes = [n for n in list(G.nodes()) if n not in (all_suburbs + all_cbds)]
    
    return all_cbds,all_suburbs,all_other_nodes

#%%

def direct_dist_between_nodes(from_node,to_node,n_c,m_c,clust_to_row,clust_to_col,cluster_map):

    c_f = from_node[0]
    m_f = from_node[1]
    n_f = from_node[2]
    c_t = to_node[0]
    m_t = to_node[1]
    n_t = to_node[2]
    
    lined_up = False
    matched = False
    count_down = 0
    count_across = 0
    dist_between_nodes = 0

    #Same Cluster
    if c_f == c_t:
        if m_f == m_t and n_f < n_t:
    #        Straight Down
            while matched == False:
                n_f += 1
                count_down += 1
                if n_f == n_t:
                    matched = True
                    dist_between_nodes = count_down
        elif m_f == m_t and n_f > n_t:
    #        Straight Up
            while matched == False:
                n_f -= 1
                count_down += 1
                if n_f == n_t:
                    matched = True
                    dist_between_nodes = count_down            
        elif n_f == n_t and m_f < m_t:
    #        Right Only
            while matched == False:
                m_f += 1
                count_across += 1
                if m_f == m_t:
                    matched = True
                    dist_between_nodes = count_across                    
        elif n_f == n_t and m_f > m_t:
    #        Left Only
            while matched == False:
                m_f -= 1
                count_across += 1
                if m_f == m_t:
                    matched = True
                    dist_between_nodes = count_across
        elif m_f < m_t and n_f < n_t:
    #        Down then right
            while lined_up == False:
                n_f += 1
                count_down += 1
                if n_f == n_t:
                    lined_up = True
            while matched == False:
                m_f += 1
                count_across += 1
                if m_f == m_t:
                    matched = True
                    dist_between_nodes = sqrt(count_down**2 + count_across**2)
        elif m_f > m_t and n_f < n_t:
    #        Down then left
            while lined_up == False:
                n_f += 1
                count_down += 1
                if n_f == n_t:
                    lined_up = True
            while matched == False:
                m_f -= 1
                count_across += 1
                if m_f == m_t:
                    matched = True
                    dist_between_nodes = sqrt(count_down**2 + count_across**2)
        elif m_f < m_t and n_f > n_t:
    #        Up then right
            while lined_up == False:
                n_f -= 1
                count_down += 1
                if n_f == n_t:
                    lined_up = True
            while matched == False:
                m_f += 1
                count_across += 1
                if m_f == m_t:
                    matched = True
                    dist_between_nodes = sqrt(count_down**2 + count_across**2)
        elif m_f > m_t and n_f > n_t:
    #        Up then left
            while lined_up == False:
                n_f -= 1
                count_down += 1
                if n_f == n_t:
                    lined_up = True
            while matched == False:
                m_f -= 1
                count_across += 1
                if m_f == m_t:
                    matched = True
                    dist_between_nodes = sqrt(count_down**2 + count_across**2)
    
    #Different CLusters
    else:
#        Different Cluster
        if clust_to_row[c_f] < clust_to_row[c_t]:
#            Go down
            while lined_up == False:
                count_down += 1
                if n_f == n_c and cluster_map[c_f]['d'] != None:
                    n_f = 1
                    c_f = cluster_map[c_f]['d']
                else:
                    n_f += 1
                if clust_to_row[c_f] == clust_to_row[c_t] and n_f == n_t:
                    lined_up = True
            if c_f == c_t:
#                Stay in cluster
                if m_f == m_t and n_f == n_t:
#                    Solution found straigh down
                    dist_between_nodes = count_down
                elif n_f == n_t and m_f < m_t:
#                    Go Right
                    while matched == False:
                        m_f += 1
                        count_across += 1
                        if m_f == m_t:
                            matched = True
                            dist_between_nodes = sqrt(count_down**2 + count_across**2)
                elif n_f == n_t and m_f > m_t:
#                    Go left
                    while matched == False:
                        m_f -= 1
                        count_across += 1
                        if m_f == m_t:
                            matched = True
                            dist_between_nodes = sqrt(count_down**2 + count_across**2)
            elif clust_to_col[c_f] < clust_to_col[c_t]:
#                Go right to different cluster
                while matched == False:
                    count_across += 1
                    if m_f == m_c and cluster_map[c_f]['r'] != None:
                        m_f = 1
                        c_f = cluster_map[c_f]['r']
                    else:
                        m_f += 1
                        
                    if c_f == c_t and m_f == m_t and n_f == n_t:
                        matched = True
                        dist_between_nodes = sqrt(count_down**2 + count_across**2)
            elif clust_to_col[c_f] > clust_to_col[c_t]:
#                Go left to different cluster
                while matched == False:
                    count_across += 1
                    if m_f == 1 and cluster_map[c_f]['l'] != None:
                        m_f = m_c
                        c_f = cluster_map[c_f]['l']
                    else:
                        m_f -= 1
                        
                    if c_f == c_t and m_f == m_t and n_f == n_t:
                        matched = True
                        dist_between_nodes = sqrt(count_down**2 + count_across**2)
        elif clust_to_row[c_f] > clust_to_row[c_t]:
#            Go up
            while lined_up == False:
                count_down += 1
                if n_f == 1 and cluster_map[c_f]['u'] != None:
                    n_f = n_c
                    c_f = cluster_map[c_f]['u']
                else:
                    n_f -= 1
                if clust_to_row[c_f] == clust_to_row[c_t] and n_f == n_t:
                    lined_up = True
            if c_f == c_t:
#                Stay in same cluster
                if m_f == m_t and n_f == n_t:
#                    Solution found straight up
                    dist_between_nodes = count_down
                elif n_f == n_t and m_f < m_t:
#                    Go right
                    while matched == False:
                        m_f += 1
                        count_across += 1
                        if m_f == m_t:
                            matched = True
                            dist_between_nodes = sqrt(count_down**2 + count_across**2)
                elif n_f == n_t and m_f > m_t:
#                    Go left
                    while matched == False:
                        m_f -= 1
                        count_across += 1
                        if m_f == m_t:
                            matched = True
                            dist_between_nodes = sqrt(count_down**2 + count_across**2)
            elif clust_to_col[c_f] < clust_to_col[c_t]:
#                Move right to new cluster
                while matched == False:
                    count_across += 1
                    if m_f == m_c and cluster_map[c_f]['r'] != None:
                        m_f = 1
                        c_f = cluster_map[c_f]['r']
                    else:
                        m_f += 1
                        
                    if c_f == c_t and m_f == m_t and n_f == n_t:
                        matched = True
                        dist_between_nodes = sqrt(count_down**2 + count_across**2)
            elif clust_to_col[c_f] > clust_to_col[c_t]:
#                Move left to new cluster
                while matched == False:
                    count_across += 1
                    if m_f == 1 and cluster_map[c_f]['l'] != None:
                        m_f = m_c
                        c_f = cluster_map[c_f]['l']
                    else:
                        m_f -= 1
                        
                    if c_f == c_t and m_f == m_t and n_f == n_t:
                        matched = True
                        dist_between_nodes = sqrt(count_down**2 + count_across**2)
        elif clust_to_col[c_f] < clust_to_col[c_t]:
#            Go right
            while lined_up == False:
                count_across += 1
                if m_f == m_c and cluster_map[c_f]['r'] != None:
                    m_f = 1
                    c_f = cluster_map[c_f]['r']
                else:
                    m_f += 1
                if clust_to_col[c_f] == clust_to_col[c_t] and m_f == m_t:
                    lined_up = True
            if c_f == c_t:
#                Stay in cluster
                if m_f == m_t and n_f == n_t:
#                    Solution found going right only
                    dist_between_nodes = count_across
                elif m_f == m_t and n_f < n_t:
#                    Go down
                    while matched == False:
                        n_f += 1
                        count_down += 1
                        if n_f == n_t:
                            matched = True
                            dist_between_nodes = sqrt(count_down**2 + count_across**2)
                elif m_f == m_t and n_f > n_t:
#                    Go up
                    while matched == False:
                        n_f -= 1
                        count_down += 1
                        if n_f == n_t:
                            matched = True
                            dist_between_nodes = sqrt(count_down**2 + count_across**2)
        elif clust_to_col[c_f] > clust_to_col[c_t]:
#            Go left
            while lined_up == False:
                count_across += 1
                if m_f == 1 and cluster_map[c_f]['l'] != None:
                    m_f = m_c
                    c_f = cluster_map[c_f]['l']
                else:
                    m_f -= 1
                if clust_to_col[c_f] == clust_to_col[c_t] and m_f == m_t:
                    lined_up = True
            if c_f == c_t:
#                Stay in same cluster
                if m_f == m_t and n_f == n_t:
#                    Solution found goin left only
                    dist_between_nodes = count_across
                elif m_f == m_t and n_f < n_t:
#                    Go down
                    while matched == False:
                        n_f += 1
                        count_down += 1
                        if n_f == n_t:
                            matched = True
                            dist_between_nodes = sqrt(count_down**2 + count_across**2)
                elif m_f == m_t and n_f > n_t:
#                    Go up
                    while matched == False:
                        n_f -= 1
                        count_down += 1
                        if n_f == n_t:
                            matched = True
                            dist_between_nodes = sqrt(count_down**2 + count_across**2)
    return dist_between_nodes

#%% Get query set

def get_query_set(t_max,query_set_size,all_suburbs,all_cbds,all_other_nodes,n_c,m_c,clust_to_row,clust_to_col,cluster_map,sub_from,cbd_from,sub_to,cbd_to,max_vehicle_length,G):
    
    mu, sigma = (t_max / 3),(t_max / 4)
        
    queries = []
    
    query_set_size_overselect = query_set_size* 3
    
    i = 1
    
    while i <= query_set_size_overselect:
        #Set up gate for time check
        q_in_t = False
        
        #Empty set for queries
        query = []
        
        from_rand = random.randint(1,10)
        to_rand = random.randint(1,10)
        
        if from_rand <= sub_from:
            from_node = random.choice(list(all_suburbs))
            query.append(from_node)
        elif from_rand <= cbd_from:
            from_node = random.choice(list(all_cbds))
            query.append(from_node)
        else:
            from_node = random.choice(list(all_other_nodes))
            query.append(from_node)
        
        if to_rand <= sub_to:
            to_node = random.choice(list(all_suburbs))
            query.append(to_node)
        elif to_rand <= cbd_to:
            to_node = random.choice(list(all_cbds))
            query.append(to_node)
        else:
            to_node = random.choice(list(all_other_nodes))
            query.append(to_node)
        
        if (from_node != to_node) and (from_node not in G.neighbors(to_node)):
            while q_in_t == False:
                q_t = np.random.normal(mu, sigma, 1)[0]
                if q_t >=0 and q_t <= t_max:
                    q_in_t = True
            query.append(round(q_t,4))
            query.append(direct_dist_between_nodes(from_node,to_node,n_c,m_c,clust_to_row,clust_to_col,cluster_map))
            query.append(random.randint(1,max_vehicle_length))
            queries.append(query)
            i += 1
    
    queries_df = pd.DataFrame(queries)
    queries_df.columns = ['from','to','t','length','vehicle size']
    queries_df = queries_df.sort_values(by=['length'])
    
    cut_off = int(query_set_size_overselect / 2)
    
    queries_short = queries_df[:cut_off].sample(n=query_set_size, random_state=1)
    queries_long = queries_df[cut_off:].sample(n=query_set_size, random_state=1)
    queries_all = queries_df.sample(n=query_set_size, random_state=1)
    
    return queries_short,queries_long,queries_all

#%% Useful Tools

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

#NOTE- Below returns best theoretical travel time in the network
def get_tt_of_path(p,eam):
    route_time = 0
    last_e = p[0]
    for e in p[1:]:
        route_time += eam.xs((last_e,e))['ff_travel_time']
        last_e = e
        
    return route_time

# Returns theoretical best path - needs to be fed
# n2n_best_lengths = pd.read_csv(exp_path+'/Learning/node_to_node_paths_lengths.csv', converters={"source": ast.literal_eval, "target": ast.literal_eval},index_col = [1,2])['path1']

def best_path(s,t,ds):
    if s == t:
        return 0
    else:
        return ds.xs((s,t))

#%% Offline Learning

def k_shortest_paths(G, source, target, k, edge_weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=edge_weight), k))
    
#%% Evaluation

def system_results_over_t(elm,eam):

    index_values = ['Total Cost','Utilisation Residual','Load Distribution']
    
    num_edges = elm.shape[0]
    t_steps = elm.shape[1]
    
    total_cost = []
    utilisation_residual = []
    load_dist = []
    
    for t in range(0,t_steps):
    
        total_cost_at_t = []
        utility_residual_at_t = []
        load_dist_at_t = []
        
        for i in range(0,num_edges):
            load = elm[t].values.tolist()[i]
            ff_cap = eam.values.tolist()[i][2]
            
            total_cost_at_t.append(load)
            
            #Compute Residual Free Flow Capacity
            if load < ff_cap:
                utility_residual_at_t.append(load / ff_cap)
            
            else:
                utility_residual_at_t.append(1)        
            
            #Calculate Load Distribution
            if load >= 1:
                load_dist_at_t.append(1)
            else:
                load_dist_at_t.append(0)
                
        total_cost.append(sum(total_cost_at_t))
        utilisation_residual.append(sum(utility_residual_at_t)/len(utility_residual_at_t))
        load_dist.append(sum(load_dist_at_t)/len(load_dist_at_t))
    
    
    cols = np.arange(1,(len(load_dist)+1))
    summary_df = pd.DataFrame(columns = cols,index = index_values)
    
    summary_df.loc['Total Cost'] = total_cost
    summary_df.loc['Utilisation Residual'] = utilisation_residual
    summary_df.loc['Load Distribution'] = load_dist
    
    return summary_df

def output_results(system_level_results,cpu_time,path_level_results,elm,epm,eam,exp_path,exp_number,alg_ind,alg_name,sort_by,base_folder):

    system_level_results.loc[alg_ind]['CPU Time'] = cpu_time
    system_level_results.loc[alg_ind]['Total Travel Time'] = path_level_results['length'].sum()
    system_level_results.loc[alg_ind]['Count Quickest Paths'] = path_level_results['best path'].value_counts()[True]
    try:
        system_level_results.loc[alg_ind]['Count Non Quickest Paths'] = path_level_results['best path'].value_counts()[False]
    except:
        system_level_results.loc[alg_ind]['Count Non Quickest Paths'] = 0
    system_level_results.loc[alg_ind]['Total Congestion Penalty'] = path_level_results['penalty'].sum()
    system_level_results.loc[alg_ind]['Free Flow Edges Traversed'] = path_level_results['number of freeflow edges'].sum()
    system_level_results.loc[alg_ind]['Congested Edges Traversed'] = path_level_results['number of congested flow edges'].sum()
    
    results_over_t = system_results_over_t(elm,eam)
    
    alg_folder = exp_path + '/Results/'+str(alg_name)+'/'+str(sort_by)+'/'
    results_folder = exp_path + '/Results/'
    
    if not os.path.exists(alg_folder):
        os.makedirs(alg_folder)
        
    # system_level_results.to_csv(results_folder + 'High Level Results - Exp ' +str(exp_number)+'.csv')
    path_level_results.to_csv(alg_folder + 'Path Level Results - Exp '+str(exp_number)+'.csv')
    results_over_t.to_csv(alg_folder + 'System Level Results Over Time - Exp '+str(exp_number)+'.csv')
    elm.to_csv(alg_folder + 'Edge Load Matrix - Exp '+str(exp_number)+'.csv')
    # epm.to_csv(alg_folder + 'Edge Path Matrix - Exp '+str(exp_number)+'.csv')
    eam.to_csv(alg_folder + 'Edge Attribute Matrix - Exp '+str(exp_number)+'.csv')
    
    #Add to full results
    
    count_quickest = path_level_results['best path'].value_counts()[True]
    
    try:
        count_not_quickest = path_level_results['best path'].value_counts()[False]
    except:
        count_not_quickest = 0    
    
    new_result=[exp_number,alg_name,sort_by,cpu_time,path_level_results['length'].sum(),count_quickest,count_not_quickest,path_level_results['penalty'].sum(),path_level_results['number of freeflow edges'].sum(),path_level_results['number of congested flow edges'].sum()]
    
    
    with open(base_folder+'Full Results.csv', 'a',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_result)
    
    return system_level_results

def output_training_results(path_level_results,elm,epm,exp_path,t):
    
    results_folder = exp_path + '/Learning/Training Data/'
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # path_level_results.to_csv(results_folder + 'Path Level Results_'+str(t)+'.csv')
    elm.to_csv(results_folder + 'elm_'+str(t)+'.csv')
    
def output_data(exp_path,G,queries,m,n,m_c,n_c,c,query_set_size,low_lim,upp_lim,cov_parm,t_max,sub_from,cbd_from,sub_to,cbd_to,max_vehicle_length):
    #Get new experiment dataset
    experiments_all = pd.read_csv(exp_path+'/Experiments.csv')
    exp_number = experiments_all.shape[0] + 1
    
    folder_name = exp_path + '/Experiment_'+str(exp_number)
    
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
    
    pickle.dump(G, open(sub_folder_data+'/graph_pickle.txt', 'wb'))
    
    # num_nodes = len(G)
    
    # file = open(sub_folder_data+'/graph.txt','w')
    # file.write(str(num_nodes) + '\n')
    
    # for u,v,a in G.edges(data=True):
    #     file.write(str(u)+' ')
    #     file.write(str(v)+' ')
    #     file.write(str(a['speed_limit'])+' ')
    #     file.write(str(a['length']) + '\n')
    
    # file.close()
    
    queries.to_csv(sub_folder_data+'/queries.csv')
    
    file = open(folder_name+'/experiment_details.txt','w')
    file.write('Experiment number ' + str(exp_number) + ' details : '+ '\n')
    file.write(str(m) + ' nodes across'+ '\n')
    file.write(str(n) + ' nodes down'+ '\n')
    if c > 1:
        file.write('Split into ' + str(c) + ' clusters of ' + str(m_c) + ' by ' + str(n_c) + '\n')
    else:
        file.write('With no clustering' + '\n')
    file.write('Query set of size : ' + str(query_set_size) + '\n')
    file.write('Max time period : ' + str(t_max) + '\n')
    file.write('Max vehicle length : ' + str(max_vehicle_length) + '\n')
    
    file.close()
    
    new_exp=[exp_number,m,n,c,query_set_size,low_lim,upp_lim,cov_parm,t_max,sub_from,cbd_from,sub_to,cbd_to,max_vehicle_length]
    
    with open(exp_path+'/Experiments.csv', 'a',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_exp)
        
    return sub_folder_data_training