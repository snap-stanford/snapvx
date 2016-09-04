import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import unittest

"""Test to ensure that invalid edges are not created
"""
class IllegalEdgeTest(BaseTest):

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
    suite = unittest.TestLoader().loadTestsFromTestCase(IllegalEdgeTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
