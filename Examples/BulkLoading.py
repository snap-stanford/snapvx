from snapvx import *

#Helper function for node objective
#Takes in a row from the CSV file, returns an optimization problem
def node_obj(data):
	x = Variable(1,name='x')
	return norm(x - float(data[0]))

#Helper function for edge objective
def laplace_reg(src, dst, data):
	return sum_squares(src['x'] - dst['x'])

#Load in Edge List to build graph with default node/edge objectives
gvx = LoadEdgeList('BulkLoadEdges.edges')

#Bulk Load node objectives:
#Takes one row of the CSV, uses that as input to node_obj
#There is also an (optional) input of specifying which nodes each row of the CSV refers to
gvx.AddNodeObjectives('BulkLoadData.csv', node_obj)

#Bulk load edge objectives for all edges
gvx.AddEdgeObjectives(laplace_reg)

gvx.Solve()
gvx.PrintSolution()
