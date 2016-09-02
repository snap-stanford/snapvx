import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import unittest

class NoEdgesTest(BaseTest):

    def test_no_edges(self):
        """ Test whether SnapVX works on a graph with no edges.
        """

        #generate an empty graph
        gvx = TGraphVX()

        #add two nodes to the graph with each node having a single variable with a single constraint
        x1 = Variable(name='x')
        gvx.AddNode(1, Objective=square(x1), Constraints=[x1 >= 10])
        x2 = Variable(name='x')
        gvx.AddNode(2, Objective=square(x2), Constraints=[x2 >= 5])

        #solve the optimisation problem
        gvx.Solve(UseADMM=True)

        #check if the optimal objective value is within 2 decimal places of the actual optimal value
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 125, places=2)

        #check if the final solution is within 2 decimal places of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), 10, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), 5, places=2)

        #solve the optimisation problem without using ADMM but using clustering
        gvx.Solve(UseADMM=False,UseClustering=True,ClusterSize=10)

        #check if the optimal objective value is within 2 decimal places of the actual optimal value
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 125, places=2)
        
        #check if the final solution is within 2 decimal places of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), 10, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), 5, places=2)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(NoEdgesTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
