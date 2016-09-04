import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import unittest

"""Clustering allows the original graph to be converted to a supergraph, with adjacent nodes clustered to form supernodes. Suite of tests
to check if this functionality behaves properly
"""
class ClusteringTest(BaseTest):

    def test_clustering(self):
        """ Test whether ADMM works when the graph is transformed to a supergraph by clustering adjacent nodes
        """

        #define the objective function as ||x_1-x_2||^2
        def laplace_reg(src, dst, data):
            obj = square(src['x'] - dst['x'])
            return (obj, [])

        np.random.seed(1)
        (num_nodes, num_edges) = (20,50)

        #generate a random graph with 20 nodes and 50 edges
        gvx = TGraphVX(GenRndGnm(PUNGraph, num_nodes, num_edges))

        #associate a variable x with each node and assign an objective function of ||x-a||^2 
        for i in range(num_nodes):
            x = Variable(1,name='x')
            a = np.random.randn(1)
            gvx.SetNodeObjective(i, square(x-a))

        #assign the laplacian regularisation function to each edge
        gvx.AddEdgeObjectives(laplace_reg)

        #solve the optimisation problem by switching clustering on
        gvx.Solve(UseADMM=True, UseClustering=True, ClusterSize=6)

        #check whether the solution is within 1 decimal place of the actual solution
        self.assertAlmostEqual(gvx.GetTotalProblemValue(), 19.5434, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(3, 'x'), -0.2992, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(10, 'x'), 0.1860, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(19, 'x'), -0.0299, places=1)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ClusteringTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
