import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *

class ClusteringTest(BaseTest):
    def laplace_reg(src, dst, data):
        obj = square(src['x'] - dst['x'])
        return (obj, [])

    np.random.seed(1)
    (num_nodes, num_edges) = (20,50)
    gvx = TGraphVX(GenRndGnm(PUNGraph, num_nodes, num_edges))
    for i in range(num_nodes):
        x = Variable(1,name='x')
        a = np.random.randn(1)
        gvx.SetNodeObjective(i, square(x-a))
    gvx.AddEdgeObjectives(laplace_reg)
    gvx.Solve(UseADMM=True, UseClustering=True, ClusterSize=6)
    self.assertAlmostEqual(gvx.GetTotalProblemValue(), 19.5434, places=1)
    self.assertAlmostEqual(gvx.GetNodeValue(3, 'x'), -0.2992, places=1)
    self.assertAlmostEqual(gvx.GetNodeValue(10, 'x'), 0.1860, places=1)
    self.assertAlmostEqual(gvx.GetNodeValue(19, 'x'), -0.0299, places=1)


#if __name__ == '__main__':
#    # unittest.main()
#    suite = unittest.TestLoader().loadTestsFromTestCase(BasicTest)
#    unittest.TextTestRunner(verbosity=2).run(suite)
