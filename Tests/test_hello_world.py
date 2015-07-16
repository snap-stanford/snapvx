# Base class for unit tests.
from base_test import BaseTest
import unittest
import numpy as np
from snapvx import *
from cvxpy import *

class BasicTest(BaseTest):

    def test_hello_world(self):
        """Test a basic graph problem.
        """
        #Create new graph
        gvx = TGraphVX()

        #Use CVXPY syntax to define a problem
        x1 = Variable(1, name='x1')
        obj = square(x1)
        #Add Node 1 with the given objective, with the constraint that x1 <= 10
        gvx.AddNode(1, Objective=obj, Constraints=[x1 <= 10])

        #Similarly, add Node 2 with objective |x2 + 3|
        x2 = Variable(1, name='x2')
        obj2 = abs(x2 + 3)
        gvx.AddNode(2, obj2, [])

        #Add an edge between the two nodes,
        #Define an objective, constraints using CVXPY syntax
        gvx.AddEdge(1, 2, Objective=square(norm(x1 - x2)), Constraints=[])

        gvx.Solve(useADMM=False) #Solve the problem
        # print gvx.PrintSolution() #Print entire solution on a node-by-node basis
        # print "x1 = ", x1.value, "; x2 = ", x2.value #Print the solutions of individual variables
        self.assertAlmostEqual(x1.value, -0.5, places=3)
        self.assertAlmostEqual(x2.value, -1, places=3)

if __name__ == '__main__':
    unittest.main()
