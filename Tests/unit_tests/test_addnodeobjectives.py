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

"""Suite of tests to check if the adding a single objective function to multiple edges behaves appropriately"""
class AddNodeObjectivesTest(BaseTest):

    DATA_DIR = '../TestData'
    
    def test_single_variable(self):
        """ Test whether AddNodeObjectives behaves correctly when a single variable is associated with each node
        """

        def objective_node_func(d):
            x = Variable(name='x')
            nid = int(d[0])
            if nid == 1:
                # Node 1 has objective (x + 0)^2, from basic_bulk_load.csv
                return square(x + int(d[1]))
            elif nid == 2:
                # Node 2 has objective |x + 3|, from basic_bulk_load.csv
                return abs(x + int(d[1]))
        
        #Load the graph from file
        gvx = LoadEdgeList(os.path.join(self.DATA_DIR, 'basic_bulk_load.edges'))
        #add the objective function
        gvx.AddNodeObjectives(os.path.join(self.DATA_DIR, 'basic_bulk_load.csv'),
                              objective_node_func)

        #Solve the optimisation problem
        gvx.Solve()
        
        #Check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), 0, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -3, places=1)

    def test_multi_variables(self):
        """ Test whether the function works in case of multiple variables
        """
        # Node 1: objective = x1^2 + |y1 + 4|
        # Node 2: objective = (x2 + 3)^2 + |y2 + 6|
        def objective_node_func(d):
            x = Variable(name='x')
            y = Variable(name='y')
            obj = square(x + int(d[0])) + abs(y + int(d[1]))
            constraints = []
            return (obj, constraints)

        # Form a graph with two nodes and an edge between them.
        gvx = LoadEdgeList(os.path.join(self.DATA_DIR, 'multi_vars.edges'))
        #add the same objective function to both nodes
        gvx.AddNodeObjectives(os.path.join(self.DATA_DIR, 'multi_vars.csv'),
                              objective_node_func,
                              NodeIDs=[1,2])

        #Solve the optimisation problem
        gvx.Solve()

        #Check if the solution is within a decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 0, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), 0, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -3, places=1)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(AddNodeObjectivesTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
