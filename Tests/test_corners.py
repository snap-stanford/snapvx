import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import unittest

class CornerTest(BaseTest):

    DATA_DIR = 'TestData'


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

        gvx.Solve(UseADMM=False,UseClustering=True,ClusterSize=10)
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
    suite = unittest.TestLoader().loadTestsFromTestCase(CornerTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
