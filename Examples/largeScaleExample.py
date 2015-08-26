from snapvx import *
import numpy
import time

def laplace_reg(src, dst, data):
    return (sum_squares(src['x'] - dst['x']), [])

num_nodes = 10
size_prob = 5000

temp = GenRndDegK(num_nodes, 3)
gvx = TGraphVX(temp)

for i in range(num_nodes):
    x = Variable(size_prob,name='x')
    a = numpy.random.randn(size_prob)
    gvx.SetNodeObjective(i, square(norm(x-a)))

gvx.AddEdgeObjectives(laplace_reg)

start = time.time()
gvx.Solve(verbose=True, rho=1.0)#1.0 vs 1.1
ellapsed = time.time() - start
print ellapsed, "seconds; with ADMM"


start = time.time()
#gvx.Solve(useADMM=False)                                                                                             
ellapsed = time.time() - start
print ellapsed, "seconds; no ADMM"

