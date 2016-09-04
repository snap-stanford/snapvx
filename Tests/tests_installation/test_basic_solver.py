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

"""Suite of tests to check if the basic functionality of snapvx is behaving correctly"""
class BasicSolverTest(BaseTest):

    DATA_DIR = os.getcwd()+'/Tests/TestData'

    def test_basic_solver_with_ADMM(self):
        """ Test a basic SnapVX graph problem with ADMM.
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

        gvx.Solve(UseADMM=True) # Solve the problem with ADMM
        
        #check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(x1.value, -0.5, places=1)
        self.assertAlmostEqual(x2.value, -1, places=1)

 
    def test_basic_solver_without_ADMM(self):
        """ Test a basic SnapVX graph problem without ADMM
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

        gvx.Solve(UseADMM=False) # Solve the problem without ADMM
        
        #check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(x1.value, -0.5, places=3)
        self.assertAlmostEqual(x2.value, -1, places=3)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(BasicSolverTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
