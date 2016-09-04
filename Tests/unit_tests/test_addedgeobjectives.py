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
class AddEdgeObjectiveTest(BaseTest):

    DATA_DIR = '../TestData'
    
 
    def test_single_variable(self):
        """ Test whether AddEdgeObjectives behaves correctly when a single variable is associated with each node
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
        
        #define the edge objective function
        def objective_edge_func(src, dst, data):
            # The edge has objective ||x1 - x2||^2
            return square(norm(src['x'] - dst['x']))
        
        #Load the graph from file
        gvx = LoadEdgeList(os.path.join(self.DATA_DIR, 'basic_bulk_load.edges'))
        #Set the node objectives
        gvx.AddNodeObjectives(os.path.join(self.DATA_DIR, 'basic_bulk_load.csv'),
                              objective_node_func)
        #Set the Edge objectives
        gvx.AddEdgeObjectives(objective_edge_func)
        
        #Solve the optimisation problem
        gvx.Solve()
        
        #Check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.5, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -1, places=1)

    def test_multi_variables(self):
        """ Test whether AddEdgeObjectives behaves correctly when a single variable is associated with each node
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
        #Set the node objectives
        gvx.AddNodeObjectives(os.path.join(self.DATA_DIR, 'multi_vars.csv'),
                              objective_node_func,
                              NodeIDs=[1,2])
        #Set the edge objectives
        gvx.AddEdgeObjectives(objective_edge_func)
        
        #Solve the optimisation problem
        gvx.Solve()

        #Check if the solution is within a decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 3.3125, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.25, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -2.75, places=1)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(AddEdgeObjectiveTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
