import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import unittest

"""Suite of tests to check if the function used for adding a node to the graph works properly"""
class AddNodeTest(BaseTest):

    def test_node_count(self):
        """ Test whether AddNode function sets the nodes correctly
        """

        #generate an empty graph
        gvx = TGraphVX()

        #add two nodes to the graph with
        gvx.AddNode(1)
        gvx.AddNode(2)
        
        #check that the node count is 2
        self.assertEqual(sum(1 for _ in gvx.Nodes()), 2)

    def test_node_objective(self):
        """Test whether adding a node objective results in correct solution
        """
        
        #generate an empty graph
        gvx = TGraphVX()

        consts = []
        #add two nodes to the graph, each with a single dimensional variable and objective as ||x-a||^2
        for i in range(2):
            x = Variable(1,name='x') #Each node has its own variable named 'x'
            a = numpy.random.randn(1)
            consts.append(a)
            gvx.AddNode(i+1, Objective=square(norm(x-a)))

        #solve the optimisation problem
        gvx.Solve(UseADMM=True)

        #check if the final solution is within 2 decimal places of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), consts[0], places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), consts[1], places=2)
    
    def test_node_constraint(self):
        """Test whether adding a node constraint results in correct solution
        """
        
        #generate an empty graph
        gvx = TGraphVX()

        consts = []
        #add two nodes to the graph, each with a single dimensional variable, objective as ||x-a||^2 and a constraint of a+1
        for i in range(2):
            x = Variable(1,name='x') #Each node has its own variable named 'x'
            a = numpy.random.randn(1)
            consts.append(a)
            gvx.AddNode(i+1, Objective=square(norm(x-a)), Constraints=[x >= a+1])

        #solve the optimisation problem
        gvx.Solve(UseADMM=True)

        #check if the final solution is within 2 decimal places of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), consts[0] + 1, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), consts[1] + 1, places=2)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(AddNodeTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
