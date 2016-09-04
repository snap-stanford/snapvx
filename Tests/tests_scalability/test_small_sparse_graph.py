import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import time
import unittest

"""Obtain snapvx solution time estimate on sparse graphs of small size"""
class SmallSparseGraphTest(BaseTest):

    DATA_DIR = 'TestData'

    def test_small_sparse_graph(self):
        """ Test solution time on small sparse graph
        """    
        var_size = 100
        np.random.seed(1)
        num_nodes = 10
        node_deg = 3
        
        # Create a random graph with 10 nodes, each with degree 3
        snapGraph = GenRndDegK(num_nodes, node_deg)
        gvx = TGraphVX(snapGraph)

        #For each node, add an objective (using random data)
        for i in range(num_nodes):
            #associate a 100 dimensional variable with each node
            x = Variable(var_size,name='x') #Each node has its own variable named 'x'
            a = numpy.random.randn(var_size)
            #set the node objective to ||x-a||^2
            gvx.SetNodeObjective(i, square(norm(x-a)))                                            
        def netLasso(src, dst, data):
            return (norm(src['x'] - dst['x'],2), [])

        #add lasso penalty for all edges
        gvx.AddEdgeObjectives(netLasso)
        start = time.time()
        gvx.Solve()
        end = time.time()
        print "Solved a problem with",num_nodes,"nodes,",node_deg,"node degree and",var_size*num_nodes,"unknowns in",end-start



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SmallSparseGraphTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
