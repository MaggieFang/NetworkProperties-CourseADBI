import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from copy import deepcopy

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def articulations(g, usegraphframe=False):
	# Get the starting count of connected components
	original_cnt = g.connectedComponents().select('component').distinct().count()

	# Default version sparkifies the connected components process 
	# and serializes node iteration.
	if usegraphframe:
		# Get vertex list for serial iteration
		vertices = g.vertices.map(lambda x: x.id).collect()

		# For each vertex, generate a new graphframe missing that vertex
		# and calculate connected component count. Then append count to
		# the output
		ans = []
		for v in vertices:
			new_vertices = g.vertices.filter('id != "' + v + '"')
			new_edges = g.edges.filter('src !="' + v + '"').filter('dst !="' + v + '"')
			new_gf = GraphFrame(new_vertices,new_edges)
			new_cnt = new_gf.connectedComponents().select('component').distinct().count()
			if new_cnt > original_cnt:
				ans.append((v,1))
			else:
				ans.append((v,0))
		return sqlContext.createDataFrame(sc.parallelize(ans),['id','articulation'])
	# Non-default version sparkifies node iteration and uses networkx 
	# for connected components count.
	else:
		gx = nx.Graph()
		gx.add_nodes_from(g.vertices.map(lambda x: x.id).collect())
		gx.add_edges_from(g.edges.map(lambda x: (x.src, x.dst)).collect())

		vertices = g.vertices.map(lambda x: x.id).collect()
		ans = []
		for v in vertices:
			tmp = deepcopy(gx)
			tmp.remove_node(v)
			new_cnt = nx.number_connected_components(tmp)
			if new_cnt > original_cnt:
				ans.append((v,1))
			else:
				ans.append((v,0))
		return sqlContext.createDataFrame(sc.parallelize(ans), ['id','articulation'])



filename = sys.argv[1]
lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs,['src','dst'])
e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness 	

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()	

# Create graphframe from the vertices and edges.
g = GraphFrame(v,e)

#Runtime approximately 5 minutes
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
df.toPandas().to_csv("articulations_out.csv")
print("---------------------------")

'''
#Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
'''