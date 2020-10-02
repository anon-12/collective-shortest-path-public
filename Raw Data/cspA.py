import pandas as pd
import osmnx as ox
import networkx as nx
import os
import pickle

# Porto
centre = (41.145556, -8.610333)
fnameIN = 'porto-data.csv'
fnameQueries = 'porto-queries.csv'
fnameShape = 'porto'
fnamePickle = 'porto-pickle.txt'

# NYC
# centre = (40.750638, -73.993899)
# fnameIN = 'nyc-data.csv'
# fnameQueries = 'nyc-queries.csv'
# fnameShape = 'nyc'
# fnamePickle = 'nyc-pickle.txt'

# Load data
path = '/Collective Shortest Paths/'
os.chdir(path)
queries = pd.read_csv(fnameIN, header=0)
nRecords = queries.shape[0]

# Get bounding box
station_box = ox.bbox_from_point(centre, distance=1500, project_utm=False, return_crs=False)
south = station_box[1]
north = station_box[0]
east = station_box[2]
west = station_box[3]

# Get graph
G = ox.graph_from_bbox(north, south, east, west, network_type='drive', retain_all=True)
G = nx.convert_node_labels_to_integers(G)
nodes, edges = ox.graph_to_gdfs(G)

# Concatenate lats and longs to save time
lats = queries.start_lat
lats = lats.append(queries.end_lat, ignore_index=True)
longs = queries.start_long
longs = longs.append(queries.end_long, ignore_index=True)

# Find nearest nodes
nearest_node_ids = ox.get_nearest_nodes(G, X=longs, Y=lats, method='balltree')

# Extract nearest node ids
start_nn = nearest_node_ids[0:nRecords]
end_nn = nearest_node_ids[nRecords:2*nRecords]

queries['start_nn'] = start_nn
queries['end_nn'] = end_nn

queries.to_csv(path_or_buf=fnameQueries, index=False)

# Pickle graph and save as shapefile
file = open(fnamePickle, 'wb')
pickle.dump(G, file)
file.close()
ox.save_graph_shapefile(G, filename=fnameShape)
