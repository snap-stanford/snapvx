import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import unittest

class NoEdgesTest(BaseTest):

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

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(NoEdgesTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
