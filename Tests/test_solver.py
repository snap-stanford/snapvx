import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import unittest
import numpy as np
import os.path
from snapvx import *
from cvxpy import *

class SolverTest(BaseTest):

    DATA_DIR = 'TestData'

    def test_basic_solver(self):
        """ Test a basic SnapVX graph problem.
        """
        # Create new graph
        gvx = TGraphVX()

        # Use CVXPY syntax to define a problem
        x1 = Variable(1, name='x1')
        obj = square(x1)
        # Add Node 1 with the given objective, with the constraint that x1 <= 10
        gvx.AddNode(1, Objective=obj, Constraints=[x1 <= 10])

        # Similarly, add Node 2 with objective |x2 + 3|
        x2 = Variable(1, name='x2')
        obj2 = abs(x2 + 3)
        gvx.AddNode(2, obj2, [])

        # Add an edge between the two nodes,
        # Define an objective, constraints using CVXPY syntax
        gvx.AddEdge(1, 2, Objective=square(norm(x1 - x2)), Constraints=[])

        gvx.Solve(UseADMM=True) # Solve the problem
        self.assertAlmostEqual(x1.value, -0.5, places=1)
        self.assertAlmostEqual(x2.value, -1, places=1)

        gvx.Solve(UseADMM=False) # Solve the problem with ADMM
        self.assertAlmostEqual(x1.value, -0.5, places=3)
        self.assertAlmostEqual(x2.value, -1, places=3)


    def test_bulk_loading(self):
        """ Test bulk loading.
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

        def objective_edge_func(src, dst, data):
            # The edge has objective ||x1 - x2||^2
            return square(norm(src['x'] - dst['x']))

        # Form a graph with two nodes and an edge between them.
        gvx = LoadEdgeList(os.path.join(self.DATA_DIR, 'basic_bulk_load.edges'))
        gvx.AddNodeObjectives(os.path.join(self.DATA_DIR, 'basic_bulk_load.csv'),
                              objective_node_func)
        gvx.AddEdgeObjectives(objective_edge_func)

        gvx.Solve(UseADMM=True) # Solve the problem
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.5, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -1, places=1)

        gvx.Solve(UseADMM=True,EpsAbs=0.00005,EpsRel=0.00005) #High precision
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.5, places=3)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -1, places=3)

        gvx.Solve(UseADMM=False) # Solve the problem with ADMM
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.5, places=3)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -1, places=3)



    def test_multi_vars(self):
        """ Test multiple variables.
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


        gvx.Solve(UseADMM=False) # Solve the problem with ADMM
        # print gvx.PrintSolution() # Print entire solution on a node-by-node basis
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 3.3125, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.25, places=3)
        # self.assertAlmostEqual(gvx.GetNodeValue(1, 'y'), -4.37, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -2.75, places=3)
        # self.assertAlmostEqual(gvx.GetNodeValue(2, 'y'), -3.23, places=1)



    def test_shared_vars_unallowed(self):
        """ Test that two nodes cannot share a variable.
        """
        gvx = TGraphVX()
        x = Variable(name='x')
        # Check that resetting an Objective is still okay.
        gvx.AddNode(1, square(x))
        gvx.SetNodeObjective(1, square(x - 2))
        try:
            # Attempt to add another node that uses the same Variable.
            gvx.AddNode(2, square(x - 1))
        except:
            # If an exception is thrown, then the test passes.
            pass
        else:
            # If an exception is not thrown, then the test fails.
            self.assertFalse(True, 'Two nodes currently share a variable')

    def test_no_edges(self):
        """ Test a graph with no edges.
        """
        gvx = TGraphVX()
        x1 = Variable(name='x')
        gvx.AddNode(1, Objective=square(x1), Constraints=[x1 >= 10])
        x2 = Variable(name='x')
        gvx.AddNode(2, Objective=square(x2), Constraints=[x2 >= 5])

        gvx.Solve(UseADMM=True)
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 125, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), 10, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), 5, places=2)


    def test_illegal_edge(self):
        """ Test that invalid edges cannot be created.
        """
        gvx = TGraphVX()
        x = Variable(name='x')
        # Check that resetting an Objective is still okay.
        gvx.AddNode(1, square(x))
        gvx.SetNodeObjective(1, square(x - 2))
        try:
            # Add an edge when one of the specified nodes does not exist.
            gvx.AddEdge(1, 2, Objective=0, Constraints=[])
        except:
            # If an exception is thrown, then the test passes.
            pass
        else:
            # If an exception is not thrown, then the test fails.
            self.assertFalse(True, 'An illegal edge was added.')

if __name__ == '__main__':
#    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(SolverTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
