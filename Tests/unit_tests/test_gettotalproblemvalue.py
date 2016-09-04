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

"""Suite of tests to check if the function for obtaining the optimal value of convergence returns correct values"""
class GetTotalProblemValueTest(BaseTest):
 
    def test_single_variable(self):
        """ Test whether the value returned for a single variable and a single node is correct
        """
        # Create new graph
        gvx = TGraphVX()

        # Use CVXPY syntax to define a problem
        x1 = Variable(1, name='x1')
        obj = square(x1)
        # Add Node 1 with the given objective, with the constraint that x1 >= 10
        gvx.AddNode(1, Objective=obj, Constraints=[x1 >= 10])

        gvx.Solve()

        #Check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 100, places=1)
    
    def test_multiple_variables(self):
        """ Test whether the value returned for two variables and a single node is correct
        """
        # Create new graph
        gvx = TGraphVX()

        # Use CVXPY syntax to define a problem
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        
        obj = square(x1) + abs(x2+3)
        # Add Node 1 with the given objective and constraints on x1 and x2
        gvx.AddNode(1, Objective=obj, Constraints=[x1 >= 10, x2 <= -5])

        gvx.Solve()

        #Check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 102, places=1)
    
    def test_multiple_nodes(self):
        """ Test whether the value returned for variables on multiple nodes is correct
        """
        # Create new graph
        gvx = TGraphVX()
 
        consts = []
        #add two nodes to the graph, each with a single dimensional variable and objective as ||x-a||^2
        for i in range(2):
            x = Variable(1,name='x') #Each node has its own variable named 'x'
            a = numpy.random.randn(1)
            consts.append(a)
            gvx.AddNode(i+1, Objective=square(norm(x-a)))
    
        gvx.Solve()

        #Check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 0, places=1)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(GetTotalProblemValueTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
