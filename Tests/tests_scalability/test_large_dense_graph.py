import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import time
import unittest

class LargeDenseGraphTest(BaseTest):

    def test_large_dense_graph(self):
        """ Test solution time on sparse graph
        """    
        var_size = 1000
        np.random.seed(1)
        num_nodes = 1000
        node_deg = 10
        # Create new graph
        snapGraph = GenRndDegK(num_nodes, node_deg)
        gvx = TGraphVX(snapGraph)

        #For each node, add an objective (using random data)
        for i in range(num_nodes):
            x = Variable(var_size,name='x') #Each node has its own variable named 'x'
            a = numpy.random.randn(var_size)
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
#    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(LargeDenseGraphTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
