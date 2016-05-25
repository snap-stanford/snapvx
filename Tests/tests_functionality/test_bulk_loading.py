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

class BulkLoadingTest(BaseTest):

    DATA_DIR = '../TestData'

    def test_bulk_loading_with_ADMM(self):
        """ Test bulk loading.
        """
        def objective_node_func(d):
            x = Variable(name='x')
            nid = int(d[0])
            if nid == 1:
                # Node 1 has objective (x + 0)^2, from basic_bulk_load.csv
                return square(x + int(d[1]))
            elif nid == 2:
                # Node 2 has objective |x + 3|, from basic_bulk_load.csv
                return abs(x + int(d[1]))

        def objective_edge_func(src, dst, data):
            # The edge has objective ||x1 - x2||^2
            return square(norm(src['x'] - dst['x']))

        # Form a graph with two nodes and an edge between them.
        gvx = LoadEdgeList(os.path.join(self.DATA_DIR, 'basic_bulk_load.edges'))
        gvx.AddNodeObjectives(os.path.join(self.DATA_DIR, 'basic_bulk_load.csv'),
                              objective_node_func)
        gvx.AddEdgeObjectives(objective_edge_func)

        gvx.Solve(UseADMM=True) # Solve the problem
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.5, places=1)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -1, places=1)
 
    def test_bulk_loading_with_ADMM_high_prec(self):
        """ Test bulk loading.
        """
        def objective_node_func(d):
            x = Variable(name='x')
            nid = int(d[0])
            if nid == 1:
                # Node 1 has objective (x + 0)^2, from basic_bulk_load.csv
                return square(x + int(d[1]))
            elif nid == 2:
                # Node 2 has objective |x + 3|, from basic_bulk_load.csv
                return abs(x + int(d[1]))

        def objective_edge_func(src, dst, data):
            # The edge has objective ||x1 - x2||^2
            return square(norm(src['x'] - dst['x']))

        # Form a graph with two nodes and an edge between them.
        gvx = LoadEdgeList(os.path.join(self.DATA_DIR, 'basic_bulk_load.edges'))
        gvx.AddNodeObjectives(os.path.join(self.DATA_DIR, 'basic_bulk_load.csv'),
                              objective_node_func)
        gvx.AddEdgeObjectives(objective_edge_func)

        gvx.Solve(UseADMM=True,EpsAbs=0.00005,EpsRel=0.00005) #High precision
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.5, places=3)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -1, places=3)
 
    def test_bulk_loading_without_ADMM(self):
        """ Test bulk loading.
        """
        def objective_node_func(d):
            x = Variable(name='x')
            nid = int(d[0])
            if nid == 1:
                # Node 1 has objective (x + 0)^2, from basic_bulk_load.csv
                return square(x + int(d[1]))
            elif nid == 2:
                # Node 2 has objective |x + 3|, from basic_bulk_load.csv
                return abs(x + int(d[1]))

        def objective_edge_func(src, dst, data):
            # The edge has objective ||x1 - x2||^2
            return square(norm(src['x'] - dst['x']))

        # Form a graph with two nodes and an edge between them.
        gvx = LoadEdgeList(os.path.join(self.DATA_DIR, 'basic_bulk_load.edges'))
        gvx.AddNodeObjectives(os.path.join(self.DATA_DIR, 'basic_bulk_load.csv'),
                              objective_node_func)
        gvx.AddEdgeObjectives(objective_edge_func)

        gvx.Solve(UseADMM=False) # Solve the problem without ADMM
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), -0.5, places=3)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), -1, places=3)




if __name__ == '__main__':
#    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(BulkLoadingTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
