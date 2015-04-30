from snapvx import *
import numpy

#Helper function to define edge objectives
#Takes in two nodes, returns a problem defined with the variables from those nodes
def laplace_reg(src, dst):
	return sum_squares(src['x'] - dst['x']) #TODO: ALSO RETURN CONSTRAINTS IN TUPLE

#Generate random graph, using SNAP syntax
numpy.random.seed(1)
num_nodes = 10
num_edges = 30
n = 10
snapGraph = GenRndGnm(PUNGraph, num_nodes, num_edges)
gvx = TGraphVX(snapGraph)

#For each node, add an objective (using random data)
for i in range(num_nodes):
	x = Variable(n,name='x')
	a = numpy.random.randn(n)
	gvx.SetNodeObjective(i, square(norm(x-a)))

#Set all edge objectives at once (Laplacian Regularization)
gvx.AddEdgeObjectives(laplace_reg)


gvx.Solve()
gvx.PrintSolution()

