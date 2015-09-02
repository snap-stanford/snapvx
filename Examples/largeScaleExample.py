from snapvx import *
import numpy
import time

def laplace_reg(src, dst, data):
    return (norm(src['x'] - dst['x']), [])

numpy.random.seed(0)
num_nodes = 1000#1000
size_prob = 9000#9000

temp = GenRndDegK(num_nodes, 3)
gvx = TGraphVX(temp)

for i in range(num_nodes):
    x = Variable(size_prob,name='x')
    a = numpy.random.randn(size_prob)
    gvx.SetNodeObjective(i, sum_entries(huber(x-a)))


gvx.AddEdgeObjectives(laplace_reg)

start = time.time()
gvx.Solve(verbose=True, rho=0.1)#0.1
ellapsed = time.time() - start
print ellapsed, "seconds; with ADMM"
gvx.PrintSolution()

start = time.time()
#gvx.Solve(useADMM=False)                                                                                             
ellapsed = time.time() - start
print ellapsed, "seconds; no ADMM"

