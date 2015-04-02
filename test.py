## snapvx Tests

from snapvx import *
import numpy
import random
import time

testADMM = True

# Simple test on a graph with 2 nodes and 1 edge.
# Node 1: objective = x1^2
# Node 2: objective = |x2 + 3|
# Edge: objective = ||x1 - x2||^2
#
# With no constraints:
# Expect optimal status with a value of 2.5
# x1 = -0.5, x2 = -1
def test1(testADMM=False):
    gvx = TGraphVX(2, 1)
    x1 = Variable()
    x2 = Variable()
    objNode1 = square(x1)
    objNode2 = abs(x2 + 3)
    gvx.AddNode(1, objNode1, x1, Constraints=[x1 >= 0])
    gvx.AddNode(2, objNode2, x2)

    # Test getting Variables via gvx
    n1Var = gvx.GetNodeVariable(1)
    n2Var = gvx.GetNodeVariable(2)
    objEdge = square(norm(n1Var - n2Var))
    gvx.AddEdge(1, 2, objEdge, Constraints=[x2 <= -2])

    # ADMM test to ensure that calculated values are the same.
    if testADMM:
        t0 = time.time()
        gvx.Solve(useADMM=True)
        t1 = time.time()
        print 'ADMM Solution [%.4f seconds]' % (t1 - t0)
        gvx.PrintSolution()
        gvx.PrintSolution('test1-ADMM.out')

    t0 = time.time()
    gvx.Solve(useADMM=False)
    t1 = time.time()
    print 'Serial Solution [%.4f seconds]' % (t1 - t0)
    print gvx.status, gvx.value
    gvx.PrintSolution()
    gvx.PrintSolution('test1-serial.out')


# Larger test on a graph with 100 nodes and approximately 300 edges.
# All vectors are in R^10.
# Each node i: objective = ||x_i - a||^2 where a is randomly generated.
# Each edge {i,j}: objective = ||x_i - x_j||^2
# No constraints.
def test2(testADMM=False):
    numpy.random.seed(1)
    random.seed(1)
    num_nodes = 100
    num_edges = 300
    n = 10
    gvx = TGraphVX(num_nodes, num_edges)

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
    testNIds = [5, 33, 41, 68, 97]

    # ADMM test to ensure that calculated values are the same.
    if testADMM:
        t0 = time.time()
        gvx.Solve(useADMM=True)
        t1 = time.time()
        print 'ADMM Solution [%.4f seconds]' % (t1 - t0)
        for nid in testNIds:
            print nid, gvx.GetNodeValue(nid)
        gvx.PrintSolution('test2-ADMM.out')

    t0 = time.time()
    gvx.Solve(useADMM=False)
    t1 = time.time()
    print 'Serial Solution [%.4f seconds]' % (t1 - t0)
    print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges()), gvx.status, gvx.value
    for nid in testNIds:
        print nid, gvx.GetNodeValue(nid)
    gvx.PrintSolution('test2-serial.out')


# Largest test on a graph with num_nodes nodes and num_edges edges.
# All vectors are in R^n.
# Each node i: objective = ||x_i - a||^2 where a is randomly generated.
# Each edge {i,j}: objective = ||x_i - x_j||^2
# No constraints.
def test3(testADMM=False):
    numpy.random.seed(1)
    random.seed(1)
    num_nodes = 1000
    num_edges = 5000
    n = 1000
    gvx = TGraphVX(num_nodes, num_edges)

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
        while nid1 == nid2 or (gvx.IsEdge(nid1, nid2)):
            nid1 = random.randint(1, num_nodes)
            nid2 = random.randint(2, num_nodes)
        x1 = gvx.GetNodeVariable(nid1)
        x2 = gvx.GetNodeVariable(nid2)
        objective = square(norm(x1 - x2))
        gvx.AddEdge(nid1, nid2, objective)
    print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges())

    # Solve and print results for sanity check.
    testNIds = [57, 246, 295, 501, 724]

    # ADMM test to ensure that calculated values are the same.
    if testADMM:
        t0 = time.time()
        gvx.Solve(useADMM=True)
        t1 = time.time()
        print 'ADMM Solution [%.4f seconds]' % (t1 - t0)
        for nid in testNIds:
            print nid, gvx.GetNodeValue(nid)
        gvx.PrintSolution('test3-ADMM.out')

    t0 = time.time()
    gvx.Solve(useADMM=False)
    t1 = time.time()
    print 'Serial Solution [%.4f seconds]' % (t1 - t0)
    print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges()), gvx.status, gvx.value
    for nid in testNIds:
        print nid, gvx.GetNodeValue(nid)
    gvx.PrintSolution('test3-serial.out')


def main():
    print '*************** TEST 1 ***************'
    test1(testADMM=testADMM)
    print '*************** TEST 2 ***************'
    # test2(testADMM=testADMM)
    print '*************** TEST 3 ***************'
    # test3(testADMM=testADMM)
    print '**************** Done ****************'

if __name__ == "__main__":
    main()
