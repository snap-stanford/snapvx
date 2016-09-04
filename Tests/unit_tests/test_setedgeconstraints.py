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

"""Suite of tests to check if the function which sets the constraint for each edge individually works correctly"""
class SetEdgeConstraintsTest(BaseTest):
 
    def test_single_edge(self):
        """ Test for function correctness in case of a single edge
        """
        # Create new graph
        gvx = TGraphVX()

        # Define two variables
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        
        # Add Node 1  with a single objective and a single constraint
        gvx.AddNode(1, Objective=abs(x1), Constraints=[x1 >= 10])
        # Add Node 2  with a single objective and a single constraint
        gvx.AddNode(2, Objective=abs(x2), Constraints=[x2 >= 10])

        #create an edge between 1 and 2
        gvx.AddEdge(1,2)

        #set edge constraint
        gvx.SetEdgeConstraints(1,2,[x1 <= 10, x2 <= 10])

        #solve the optimisation problem
        gvx.Solve()

        #Check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x1'), 10, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x2'), 10, places=1)
    
    def test_multiple_edges(self):
        """ Test for function correctness in case of multiple edges
        """
        # Create new graph
        gvx = TGraphVX()

        # Define three variables
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x3 = Variable(1, name='x3')
        
        # Add Node 1  with a single objective and a single constraint
        gvx.AddNode(1, Objective=abs(x1), Constraints=[x1 >= 10])
        # Add Node 2  with a single objective and a single constraint
        gvx.AddNode(2, Objective=abs(x2), Constraints=[x2 >= 10])
        # Add Node 3  with a single objective and a single constraint
        gvx.AddNode(3, Objective=abs(x3), Constraints=[x3 >= 10])

        #create an edge between 1 and 2
        gvx.AddEdge(1,2)
        #create an edge between 2 and 3
        gvx.AddEdge(2,3)

        #set edge constraints for the two edges separately
        gvx.SetEdgeConstraints(1,2,[x1 <= 10, x2 <= 10])
        gvx.SetEdgeConstraints(2,3,[x2 <= 10, x3 <= 10])

        #solve the optimisation problem
        gvx.Solve()

        #Check if the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x1'), 10, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x2'), 10, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(3, 'x3'), 10, places=1)
    

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SetEdgeConstraintsTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
