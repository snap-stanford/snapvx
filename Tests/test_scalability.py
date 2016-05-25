import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import time

class ScalabilityTest(BaseTest):

    DATA_DIR = 'TestData'
    var_size = 5

    def test_sparse_graph(self):
        """ Test solution time on sparse graph
        """    
        np.random.seed(1)
        num_nodes = 1000
        num_edges = 3000
        # Create new graph
        snapGraph = GenRndGnm(PUNGraph, num_nodes, num_edges)
        gvx = TGraphVX(snapGraph)

        #For each node, add an objective (using random data)
        for i in range(num_nodes):
            x = Variable(var_size,name='x') #Each node has its own variable named 'x'
            a = numpy.random.randn(n)
            gvx.SetNodeObjective(i, square(norm(x-a)))                                            
            def netLasso(src, dst, data):
                    return (norm(src['x'] - dst['x'],2), [])

        #add lasso penalty for all edges
        gvx.AddEdgeObjectives(netLasso,ObjFunx = lambda src,dst,data:(norm(src['x'] - dst['x'],2), []))
        start = time.time()
        gvx.Solve()
        end = time.time()
        print "Solved a problem with",num_nodes,"nodes,",num_edges,"edges and",var_size,"unknowns in",end-start

    def test_dense_graph(self):
        """ Test solution time on dense graph
        """    
        np.random.seed(1)
        num_nodes = 50
        num_edges = 2000
        # Create new graph
        snapGraph = GenRndGnm(PUNGraph, num_nodes, num_edges)
        gvx = TGraphVX(snapGraph)

        #For each node, add an objective (using random data)
        for i in range(num_nodes):
            x = Variable(var_size,name='x') #Each node has its own variable named 'x'
            a = numpy.random.randn(n)
            gvx.SetNodeObjective(i, square(norm(x-a)))                                            
            def netLasso(src, dst, data):
                    return (norm(src['x'] - dst['x'],2), [])

        #add lasso penalty for all edges
        gvx.AddEdgeObjectives(netLasso,ObjFunx = lambda src,dst,data:(norm(src['x'] - dst['x'],2), []))
        start = time.time()
        gvx.Solve()
        end = time.time()
        print "Solved a problem with",num_nodes,"nodes,",num_edges,"edges and",var_size,"unknowns in",end-start



#if __name__ == '__main__':
#    # unittest.main()
#    suite = unittest.TestLoader().loadTestsFromTestCase(BasicTest)
#    unittest.TextTestRunner(verbosity=2).run(suite)
