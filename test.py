## snapvx Tests

from snapvx import *
import numpy
import random


# Simple test on a graph with 2 nodes and 1 edge.
# Node 1: objective = x1^2
# Node 2: objective = |x2 + 3|
# Edge: objective = ||x1 - x2||^2
# No constraints.
# Expect optimal status with a value of 2.5
# x1 = -0.5, x2 = -1
def test1():
    gvx = TUNGraphVX(2, 1)
    x1 = Variable()
    x2 = Variable()
    objNode1 = square(x1)
    objNode2 = abs(x2 + 3)
    gvx.AddNode(1, objNode1, x1)
    gvx.AddNode(2, objNode2, x2)

    # Test getting Variables via gvx
    n1Var = gvx.GetNodeVariable(1)
    n2Var = gvx.GetNodeVariable(2)
    objEdge = square(norm(n1Var - n2Var))
    gvx.AddEdge(1, 2, objEdge)

    gvx.Solve()
    print gvx.status, gvx.value
    print x1.value, x2.value


# Larger test on a graph with 100 nodes and approximately 300 edges.
# All vectors are in R^10.
# Each node i: objective = ||x_i - a||^2 where a is randomly generated.
# Each edge {i,j}: objective = ||x_i - x_j||^2
# No constraints.
def test2():
    numpy.random.seed(1)
    random.seed(1)
    num_nodes = 100
    num_edges = 300
    n = 10
    gvx = TUNGraphVX(num_nodes, num_edges)

    # Add nodes to graph.
    for i in xrange(1, num_nodes + 1):
        x = Variable(n)
        a = numpy.random.randn(n)
        objective = square(norm(x - a))
        gvx.AddNode(i, objective, x)

    # Add edges to graph by choosing two random nids. If nids are equal or
    # the edge already exists, skip.
    for i in xrange(1, num_edges + 1):
        nid1 = random.randint(1, num_nodes)
        nid2 = random.randint(2, num_nodes)
        if nid1 == nid2 or (gvx.IsEdge(nid1, nid2)):
            continue
        x1 = gvx.GetNodeVariable(nid1)
        x2 = gvx.GetNodeVariable(nid2)
        objective = square(norm(x1 - x2))
        gvx.AddEdge(nid1, nid2, objective)

    # Solve and print results for sanity check.
    gvx.Solve()
    print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges()), gvx.status, gvx.value
    for i in xrange(5):
        nid = random.randint(1, num_nodes)
        x = gvx.GetNodeVariable(nid)
        print nid, x.value


def main():
    print '** TEST 1 **'
    test1()
    print '** TEST 2**'
    test2()
    print '** Done **'


if __name__ == "__main__":
    main()
