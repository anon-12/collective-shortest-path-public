Part 1 - Raw Data Collection and Processing:

The real world data (e.g. taxi OD journeys) can be downloaded from:

Porto: http://www.geolink.pt/ecmlpkdd2015-challenge/dataset.html
New York: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Once downloaded place in folder "raw data"

-Initially run the code "extractData.m" in order to clean the raw data and apply some basic spatiotemporal filters on the raw query data.
-The output is then read in using code "cspA.py" to extract the road network using OSMNX module, and to link the latitude/longitude co-ordinates of the query set to their nearest node on the extracted network.
-Next run edgeEditingA.M in order to assign speed limit to each edge in the graph

The output of these processes will the set of data as input to the end-to-end experiments: 
-nyc-pickle.txt/porto-pickle.txt : pickle of networkx graph object
-nyc-edge-info.csv/porto-edge-info.csv : edge level information (e.g. speed limit, length)
-nyc-queries.csv/porto-queries.csv : navigational queries (with their nearest node on the graph and origin and destination)


Part 2 - Running Experiments

- The codes in folder "Code" can be run in numerical order - the codes are commented and their purposes listed in the comments.
- Each code references "csp_algorithms" and "csp_toolkit" which contains all the functions which drive the CSP experiment code base.
- The path references need to be updated and these are marked in the code as to where they need to be directed to.
- All of the codes from 2.1 onwards require "Experiment Parameters" - this will be either: "NY", "Porto" or "Synthetic" and the experinment number which will be determined by the folder structures created in codes 1.1 and 1.2.
- Each experiment has it's own folder stucture - these are references automatically throughout all of the codes:
	-Data
		-Training Data
	-Learning
		-Training Data
	-Results
		-Seperate folder for each algorithm variant that is run
- In the high level folder for each "world" (e.g. synthetic, New York or Porto) results summaries are stored for all the experiment that are run, this is updated automatically.
- It is recommended that the experiments are set up to run on a linux server, and for the paralized version ensure that you have the capacilities.
- All the results are automatically output

Python 3.7 has been used throughout (except for the three matlab codes used in initial set up). The following python modules are required:
-pandas
-pickle
-ast
-sklearn
-tensorflow
-networkx
-osmnx
-h5py
-multiprocessing
