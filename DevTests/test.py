## snapvx Tests

import sys
sys.path.append('..')

from snapvx import *
import numpy
import random
import time

testADMM = True

def main():
    # test1(testADMM=testADMM)
    # test2(testADMM=testADMM)
    # test3(testADMM=testADMM)
    # test4(testADMM=testADMM)
    # test5(testADMM=testADMM)
    # test6(testADMM=testADMM)
    # test7(testADMM=testADMM)
    print '**************** DONE *****************'

def printTest(num):
    s = str(num)
    if num < 10: s = '0' + s
    print '*************** TEST %s ***************' % s


# Simple test on a graph with 2 nodes and 1 edge.
# Node 1: objective = x1^2
# Node 2: objective = |x2 + 3|
# Edge: objective = ||x1 - x2||^2
#
# With no constraints:
# Expect optimal status with a value of 2.5
# x1 = -0.5, x2 = -1
def test1(testADMM=False):
    printTest(1)
    gvx = TGraphVX()
    x1 = Variable(name='x')
    x2 = Variable(name='x')
    objNode1 = square(x1)
    objNode2 = abs(x2 + 3)
    gvx.AddNode(1, objNode1)
    gvx.AddNode(2, objNode2)

    objEdge = square(norm(x1 - x2))
    gvx.AddEdge(1, 2, Objective=objEdge)

    # ADMM test to ensure that calculated values are the same.
    if testADMM:
        t0 = time.time()
        gvx.Solve(UseADMM=True)
        t1 = time.time()
        print 'ADMM Solution [%.4f seconds]' % (t1 - t0)
        gvx.PrintSolution()
        gvx.PrintSolution('test1-ADMM.out')

    t0 = time.time()
    gvx.Solve(UseADMM=False)
    t1 = time.time()
    print 'Serial Solution [%.4f seconds]' % (t1 - t0)
    gvx.PrintSolution()
    gvx.PrintSolution('test1-serial.out')


# Larger test on a graph with 100 nodes and approximately 300 edges.
# All vectors are in R^10.
# Each node i: objective = ||x_i - a||^2 where a is randomly generated.
# Each edge {i,j}: objective = ||x_i - x_j||^2
# No constraints.
def test2(testADMM=False):
    printTest(2)
    numpy.random.seed(1)
    random.seed(1)
    num_nodes = 100
    num_edges = 300
    n = 10
    gvx = TGraphVX()

    # Add nodes to graph.
    for i in xrange(1, num_nodes + 1):
        x = Variable(n, name='x')
        a = numpy.random.randn(n)
        objective = square(norm(x - a))
        gvx.AddNode(i, objective)

    # Add edges to graph by choosing two random nids. If nids are equal or
    # the edge already exists, skip.
    for i in xrange(1, num_edges + 1):
        nid1 = random.randint(1, num_nodes)
        nid2 = random.randint(2, num_nodes)
        if nid1 == nid2 or (gvx.IsEdge(nid1, nid2)):
            continue
        x1 = gvx.GetNodeVariables(nid1)['x']
        x2 = gvx.GetNodeVariables(nid2)['x']
        objective = square(norm(x1 - x2))
        gvx.AddEdge(nid1, nid2, Objective=objective)

    # Solve and print results for sanity check.
    testNIds = [5, 33, 41, 68, 97]

    # ADMM test to ensure that calculated values are the same.
    if testADMM:
        t0 = time.time()
        gvx.Solve(UseADMM=True)
        t1 = time.time()
        print 'ADMM Solution [%.4f seconds]' % (t1 - t0)
        print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges()), gvx.status, gvx.value
        for nid in testNIds:
            print nid, gvx.GetNodeValue(nid, 'x')
        gvx.PrintSolution('test2-ADMM.out')

    t0 = time.time()
    gvx.Solve(UseADMM=False)
    t1 = time.time()
    print 'Serial Solution [%.4f seconds]' % (t1 - t0)
    print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges()), gvx.status, gvx.value
    for nid in testNIds:
        print nid, gvx.GetNodeValue(nid, 'x')
    gvx.PrintSolution('test2-serial.out')


# Largest test on a graph with num_nodes nodes and num_edges edges.
# All vectors are in R^n.
# Each node i: objective = ||x_i - a||^2 where a is randomly generated.
# Each edge {i,j}: objective = ||x_i - x_j||^2
# No constraints.
def test3(testADMM=False):
    printTest(3)
    numpy.random.seed(1)
    random.seed(1)
    num_nodes = 1000
    num_edges = 5000
    n = 1000
    gvx = TGraphVX()

    # Add nodes to graph.
    for i in xrange(1, num_nodes + 1):
        x = Variable(n, name='x')
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
        x1 = gvx.GetNodeVariables(nid1)['x']
        x2 = gvx.GetNodeVariables(nid2)['x']
        objective = square(norm(x1 - x2))
        gvx.AddEdge(nid1, nid2, Objective=objective)
    print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges())

    # Solve and print results for sanity check.
    testNIds = [57, 246, 295, 501, 724]

    # ADMM test to ensure that calculated values are the same.
    if testADMM:
        t0 = time.time()
        gvx.Solve(UseADMM=True)
        t1 = time.time()
        print 'ADMM Solution [%.4f seconds]' % (t1 - t0)
        print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges()), gvx.status, gvx.value
        for nid in testNIds:
            print nid, gvx.GetNodeValue(nid, 'x')
        gvx.PrintSolution('test3-ADMM.out')

    t0 = time.time()
    gvx.Solve(UseADMM=False)
    t1 = time.time()
    print 'Serial Solution [%.4f seconds]' % (t1 - t0)
    print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges()), gvx.status, gvx.value
    for nid in testNIds:
        print nid, gvx.GetNodeValue(nid, 'x')
    gvx.PrintSolution('test3-serial.out')


# Simple test on a graph with 2 nodes and 1 edge using bulk loading.
# Node 1: objective = x1^2
# Node 2: objective = |x2 + 3|
# Edge: objective = ||x1 - x2||^2
#
# With no constraints:
# Expect optimal status with a value of 2.5
# x1 = -0.5, x2 = -1
def test4(testADMM=False):
    printTest(4)
    gvx = LoadEdgeList('test4.edges')
    gvx.AddNodeObjectives('test4.csv', objective_node_func_4)
    gvx.AddEdgeObjectives(objective_edge_func_4)

    # ADMM test to ensure that calculated values are the same.
    if testADMM:
        t0 = time.time()
        gvx.Solve(UseADMM=True)
        t1 = time.time()
        print 'ADMM Solution [%.4f seconds]' % (t1 - t0)
        gvx.PrintSolution()
        gvx.PrintSolution('test4-ADMM.out')

    t0 = time.time()
    gvx.Solve(UseADMM=False)
    t1 = time.time()
    print 'Serial Solution [%.4f seconds]' % (t1 - t0)
    gvx.PrintSolution()
    gvx.PrintSolution('test4-serial.out')

def objective_node_func_4(d):
    x = Variable(name='x')
    nid = int(d[0])
    if nid == 1:
        return square(x + int(d[1]))
    elif nid == 2:
        return abs(x + int(d[1]))

def objective_edge_func_4(src, dst, data):
    return square(norm(src['x'] - dst['x']))


# Simple test on a graph with 2 nodes and 1 edge using bulk loading and
# multiple variables
# Node 1: objective = x1^2 + |y1 + 4|
# Node 2: objective = (x2 + 3)^2 + |y2 + 6|
# Edge: objective = ||x1 - 2*y1 + x2 - 2*y2||^2
# Constraints: y1 > -5, y2 > -100
#
# Expect optimal status with a value of 3.3125
# x1 = -0.25, y1 = -4.67, x2 = -2.75, y2 = -3.55
def test5(testADMM=False):
    printTest(5)
    gvx = LoadEdgeList('test5.edges')
    gvx.AddNodeObjectives('test5.csv', objective_node_func_5, NodeIDs=[1,2])
    gvx.AddEdgeObjectives(objective_edge_func_5)

    # ADMM test to ensure that calculated values are the same.
    if testADMM:
        t0 = time.time()
        gvx.Solve(UseADMM=True)
        t1 = time.time()
        print 'ADMM Solution [%.4f seconds]' % (t1 - t0)
        gvx.PrintSolution()
        gvx.PrintSolution('test5-ADMM.out')

    t0 = time.time()
    gvx.Solve(UseADMM=False)
    t1 = time.time()
    print 'Serial Solution [%.4f seconds]' % (t1 - t0)
    gvx.PrintSolution()
    gvx.PrintSolution('test5-serial.out')

def objective_node_func_5(d):
    x = Variable(name='x')
    y = Variable(name='y')
    obj = square(x + int(d[0])) + abs(y + int(d[1]))
    constraints = []
    return (obj, constraints)

def objective_edge_func_5(src, dst, data):
    return square(norm(src['x'] - dst['x'] + 2 * src['y'] - 2 * dst['y']))


# Medium test on a graph with 10 nodes and approximately 30 edges.
# All vectors are in R^2.
# Each node i: objective = ||x_i - a||^2 where a is randomly generated.
# Each edge {i,j}: objective = ||x_i - x_j||^2
# No constraints.
def test6(testADMM=False):
    printTest(6)
    numpy.random.seed(10)
    random.seed(10)
    num_nodes = 10
    num_edges = 30
    n = 2
    gvx = TGraphVX()

    # Add nodes to graph.
    for i in xrange(1, num_nodes + 1):
        x = Variable(n, name='x')
        a = numpy.random.randn(n)
        objective = square(norm(x - a))
        gvx.AddNode(i, objective)

    # Add edges to graph by choosing two random nids. If nids are equal or
    # the edge already exists, skip.
    for i in xrange(1, num_edges + 1):
        nid1 = random.randint(1, num_nodes)
        nid2 = random.randint(2, num_nodes)
        if nid1 == nid2 or (gvx.IsEdge(nid1, nid2)):
            continue
        gvx.AddEdge(nid1, nid2, ObjectiveFunc=objective_edge_func_6)

    # Solve and print results for sanity check.
    testNIds = [5, 8]

    # ADMM test to ensure that calculated values are the same.
    if testADMM:
        t0 = time.time()
        gvx.Solve(UseADMM=True)
        t1 = time.time()
        print 'ADMM Solution [%.4f seconds]' % (t1 - t0)
        print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges()), gvx.status, gvx.value
        for nid in testNIds:
            print nid, gvx.GetNodeValue(nid, 'x')
        gvx.PrintSolution('test6-ADMM.out')

    t0 = time.time()
    gvx.Solve(UseADMM=False)
    t1 = time.time()
    print 'Serial Solution [%.4f seconds]' % (t1 - t0)
    print 'G(%d,%d)' % (gvx.GetNodes(), gvx.GetEdges()), gvx.status, gvx.value
    for nid in testNIds:
        print nid, gvx.GetNodeValue(nid, 'x')
    gvx.PrintSolution('test6-serial.out')

def objective_edge_func_6(src, dst):
    return square(norm(src['x'] - dst['x']))


# Simple test on a graph with 3 nodes and 2 edges using edge bulk loading
# Node i: (x_i - data)^2
# Edge (i,j): objective = ||x_i - x_j + data||^2
def test7(testADMM=False):
    printTest(7)
    gvx = LoadEdgeList('test7.edges')
    gvx.AddNodeObjectives('test7-nodes.csv', objective_node_func_7)
    gvx.AddEdgeObjectives(objective_edge_func_7, Filename='test7-edges.csv')

    # ADMM test to ensure that calculated values are the same.
    if testADMM:
        t0 = time.time()
        gvx.Solve(UseADMM=True)
        t1 = time.time()
        print 'ADMM Solution [%.4f seconds]' % (t1 - t0)
        gvx.PrintSolution()
        gvx.PrintSolution('test7-ADMM.out')

    t0 = time.time()
    gvx.Solve(UseADMM=False)
    t1 = time.time()
    print 'Serial Solution [%.4f seconds]' % (t1 - t0)
    gvx.PrintSolution()
    gvx.PrintSolution('test7-serial.out')

def objective_node_func_7(d):
    x = Variable(name='x')
    obj = square(x - int(d[0]))
    constraints = []
    return (obj, constraints)

def objective_edge_func_7(src, dst, data):
    return square(norm(src['x'] - dst['x'] + int(data[2])))


if __name__ == "__main__":
    main()
