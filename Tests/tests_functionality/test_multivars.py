import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import unittest
import numpy as np
import os.path
import os
from snapvx import *
from cvxpy import *

class MultiVarsTest(BaseTest):

    DATA_DIR = '../TestData'

    def test_multi_vars_with_ADMM(self):
        """ Test multiple variables with ADMM.
        """
        # Node 1: objective = x1^2 + |y1 + 4|
        # Node 2: objective = (x2 + 3)^2 + |y2 + 6|
        def objective_node_func(d):
            x = Variable(name='x')
            y = Variable(name='y')
            obj = square(x + int(d[0])) + abs(y + int(d[1]))
            constraints = []
            return (obj, constraints)

        # Edge: objective = ||x1 - 2*y1 + x2 - 2*y2||^2
        def objective_edge_func(src, dst, data):
            return square(norm(src['x'] - dst['x'] + 2 * src['y'] - 2 * dst['y']))

        # Form a graph with two nodes and an edge between them.
        gvx = LoadEdgeList(os.path.join(self.DATA_DIR, 'multi_vars.edges'))
        gvx.AddNodeObjectives(os.path.join(self.DATA_DIR, 'multi_vars.csv'),
                              objective_node_func,
                              NodeIDs=[1,2])
        gvx.AddEdgeObjectives(objective_edge_func)

        # The optimal solution has two different convergence points.
        # In particular, the 'y' variables at each node are different depending
        # on the Solve() method.

        gvx.Solve(UseADMM=True) # Solve the problem
        #print gvx.PrintSolution() # Print entire solution on a node-by-node basis
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 3.3125, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.25, places=1)
        # self.assertAlmostEqual(gvx.GetNodeValue(1, 'y'), -5.5625, places=3)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -2.75, places=1)
        # self.assertAlmostEqual(gvx.GetNodeValue(2, 'y'), -4.4375, places=3)


    def test_multi_vars_without_ADMM(self):
        """ Test multiple variables without ADMM.
        """
        # Node 1: objective = x1^2 + |y1 + 4|
        # Node 2: objective = (x2 + 3)^2 + |y2 + 6|
        def objective_node_func(d):
            x = Variable(name='x')
            y = Variable(name='y')
            obj = square(x + int(d[0])) + abs(y + int(d[1]))
            constraints = []
            return (obj, constraints)

        # Edge: objective = ||x1 - 2*y1 + x2 - 2*y2||^2
        def objective_edge_func(src, dst, data):
            return square(norm(src['x'] - dst['x'] + 2 * src['y'] - 2 * dst['y']))

        # Form a graph with two nodes and an edge between them.
        gvx = LoadEdgeList(os.path.join(self.DATA_DIR, 'multi_vars.edges'))
        gvx.AddNodeObjectives(os.path.join(self.DATA_DIR, 'multi_vars.csv'),
                              objective_node_func,
                              NodeIDs=[1,2])
        gvx.AddEdgeObjectives(objective_edge_func)

        # The optimal solution has two different convergence points.
        # In particular, the 'y' variables at each node are different depending
        # on the Solve() method.

        gvx.Solve(UseADMM=False) # Solve the problem with ADMM
        # print gvx.PrintSolution() # Print entire solution on a node-by-node basis
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 3.3125, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.25, places=3)
        # self.assertAlmostEqual(gvx.GetNodeValue(1, 'y'), -4.37, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -2.75, places=3)
        # self.assertAlmostEqual(gvx.GetNodeValue(2, 'y'), -3.23, places=1)



if __name__ == '__main__':
#    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(MultiVarsTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
