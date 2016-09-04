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

"""Suite of tests to check if the function which sets the objective for each node individually works correctly"""
class SetNodeObjectiveTest(BaseTest):
 
    def test_single_variable(self):
        """ Test for function correctness when a single variable is associated with a single node
        """
        # Create new graph
        gvx = TGraphVX()

        # Use CVXPY syntax to define a problem
        x1 = Variable(1, name='x1')
        obj = square(x1)

        # Add Node 1  with the constraint that x1 >= 10
        gvx.AddNode(1, Constraints=[x1 >= 10])

        #set node objective
        gvx.SetNodeObjective(1,obj)

        #solve the optimisation problem
        gvx.Solve()

        #Check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x1'), 10, places=1)
    
    def test_multiple_variables(self):
        """ Test for function correctness when multiple variables are associated with a single node
        """
        # Create new graph
        gvx = TGraphVX()

        # Use CVXPY syntax to define a problem
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        
        obj = square(x1) + abs(x2+3)
        # Add Node 1 with constraints on x1 and x2
        gvx.AddNode(1, Constraints=[x1 >= 10, x2 <= -5])

        #set node objective
        gvx.SetNodeObjective(1,obj)
        
        #solve the optimisation problem
        gvx.Solve()

        #Check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x1'), 10, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x2'), -5, places=1)
    
    def test_multiple_nodes(self):
        """ Test for function correctness when multiple nodes each contain a single variable
        """
        # Create new graph
        gvx = TGraphVX()
 
        consts = []
        #add two nodes to the graph, each with a single dimensional variable and objective as ||x-a||^2
        for i in range(2):
            x = Variable(1,name='x') #Each node has its own variable named 'x'
            a = numpy.random.randn(1)
            consts.append(a)
            #add the node to the graph and set the node objetive function
            gvx.AddNode(i+1)
            gvx.SetNodeObjective(i+1,square(norm(x-a)))
    
        #solve the optimisation problem
        gvx.Solve()

        #Check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), consts[0], places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), consts[1], places=1)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SetNodeObjectiveTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
