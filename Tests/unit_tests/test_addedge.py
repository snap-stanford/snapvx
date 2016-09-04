import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import unittest

"""Suite of tests to check if the function used for adding an edge to the graph works properly"""
class AddEdgeTest(BaseTest):

    def test_edge_count(self):
        """ Test whether AddEdge function sets the edges correctly
        """

        #generate an empty graph
        gvx = TGraphVX()

        #add three nodes
        gvx.AddNode(1)
        gvx.AddNode(2)
        gvx.AddNode(3)

        #create a dense graph
        gvx.AddEdge(1, 2)
        gvx.AddEdge(2, 3)
        gvx.AddEdge(1, 3)
        
        #check that the edge count is 3
        self.assertEqual(sum(1 for _ in gvx.Edges()), 3)

    def test_edge_objective(self):
        """Test whether adding a edge objective results in correct solution
        """
        
        #generate an empty graph
        gvx = TGraphVX()

        #add three nodes each with a single dimensional variable
        variables = []
        for i in xrange(3):
            x = Variable(1,name='x') #Each node has its own variable named 'x'
            variables.append(x)
            gvx.AddNode(i+1, Objective=square(norm(x)))
        
        #create a dense graph with an objective of ||x_1-x_2||^2 on each edge
        gvx.AddEdge(1, 2, Objective=square(norm(variables[0] - variables[1])))
        gvx.AddEdge(2, 3, Objective=square(norm(variables[1] - variables[2])))
        gvx.AddEdge(1, 3, Objective=square(norm(variables[0] - variables[2])))
        gvx.Solve(UseADMM=True)

        #check if the final solution is within 2 decimal places of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), 0, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), 0, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), 0, places=2)
    
    def test_edge_constraint(self):
        """Test whether adding edge constraints results in correct solution
        """
        
        #generate an empty graph
        gvx = TGraphVX()

        #add three nodes each with a single dimensional variable
        variables = []
        for i in xrange(3):
            x = Variable(1,name='x') #Each node has its own variable named 'x'
            variables.append(x)
            gvx.AddNode(i+1, Objective=square(norm(x)))
        
        #create a dense graph with an objective of ||x_1-x_2||^2 on each edge and a constraint
        gvx.AddEdge(1, 2, Objective=square(norm(variables[0] - variables[1])), Constraints = [variables[0] >= 2.5])
        gvx.AddEdge(2, 3, Objective=square(norm(variables[1] - variables[2])), Constraints = [variables[1] >= 2.5])
        gvx.AddEdge(1, 3, Objective=square(norm(variables[0] - variables[2])), Constraints = [variables[2] >= 2.5])
        gvx.Solve(UseADMM=True)

        #check if the final solution is within 2 decimal places of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), 2.5, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), 2.5, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), 2.5, places=2)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(AddEdgeTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
