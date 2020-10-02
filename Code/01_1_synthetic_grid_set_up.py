"""
1.1 Synthetic Grid Set Up

Purpose of code: To output a set of synthetic grids and query as per the input parameters files "syntheticParams.csv"
"""

import os
import sys


#Repository Path
path = ""
os.chdir(path)
#Path at which to land data
base_folder = ""


from csp_toolset import *

# Read in paramters

parameters = pd.read_csv('parameters.csv',index_col = 0).to_dict()
t_dom = parameters['Value']['t_dom']
max_q_t = int(((1/t_dom) * 4 - 1))
t_max = int(((1/t_dom) * 5 -1))
t_dom_seconds = int(t_dom * 3600)

#%% Experiment parameters

exp_params = pd.read_csv(base_folder + '/syntheticParams.csv')

for exp_num in range(0,exp_params.shape[0]):
    
    print('Experiment : ' + str(exp_num + 1))
    params = exp_params.loc[exp_num]
    
    m = params['M']
    n = params['N']
    c = params['clust']
    query_set_size = params['numQueries']
    query_length = params['queryLength']

    G,m_c,n_c = create_graph(m,n,c,t_dom_seconds)

    low_lim = 0.33
    upp_lim = 0.5

    G, nodes_for_bridges = add_bridges(G,c,m_c,n_c,low_lim,upp_lim,t_dom_seconds)

    cluster_map,clust_to_row,clust_to_col = create_cluster_mapping(c)

    all_cbds = []
    all_suburbs = []
    
    #Assign CBDs and Suburbs
    
    cov_parm = 0.25
    all_cbds,all_suburbs,all_other_nodes = cbds_suburbs(G,c,nodes_for_bridges,all_cbds,all_suburbs,cov_parm)
    
    #Query Sets - Actual
    
    sub_from = 6
    cbd_from = 7
    sub_to = 1
    cbd_to = 7
    max_vehicle_length = 1
    
    if query_set_size == 0:
        # num_queries = 5000
        num_queries = 20000
    elif query_set_size == 1:
        # num_queries = 10000
        num_queries = 50000
    else:
        # num_queries = 20000
        num_queries = 100000
    
    queries_short,queries_long,queries_all = get_query_set(max_q_t,num_queries,all_suburbs,all_cbds,all_other_nodes,n_c,m_c,clust_to_row,clust_to_col,cluster_map,sub_from,cbd_from,sub_to,cbd_to,max_vehicle_length,G)
    
    if query_length == 0:
        queries = queries_all
    elif query_length == 1:
        queries = queries_short
    else:
        queries = queries_long
    
    query_set_names = []
    
    for i in range(1,11):
        globals()['queries_short_'+str(i)],globals()['queries_long_'+str(i)],globals()['queries_all_'+str(i)] = get_query_set(max_q_t,num_queries,all_suburbs,all_cbds,all_other_nodes,n_c,m_c,clust_to_row,clust_to_col,cluster_map,sub_from,cbd_from,sub_to,cbd_to,max_vehicle_length,G)
        if query_length == 0:
            query_set_names.append('queries_all_'+str(i))
        elif query_length == 1:
            query_set_names.append('queries_short_'+str(i))
        else:
            query_set_names.append('queries_long_'+str(i))
    
    sub_folder_data_training = output_data(base_folder,G,queries,m,n,m_c,n_c,c,num_queries,low_lim,upp_lim,cov_parm,max_q_t,sub_from,cbd_from,sub_to,cbd_to,max_vehicle_length)
    
    for i in range(0,len(query_set_names)):
        globals()[query_set_names[i]].to_csv(sub_folder_data_training+'queries_'+str(i+1)+'.csv')

