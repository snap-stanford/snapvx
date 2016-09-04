import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import unittest

"""Test if updating rho using a custom function works"""
class SetRhoUpdateTest(BaseTest):

    def test_rho_identity(self):
        """ Test whether SnapVX works when rho is set to a constant
        """

        #generate an empty graph
        gvx = TGraphVX()

        #add two nodes to the graph with each node having a single variable with a single constraint
        x1 = Variable(name='x')
        gvx.AddNode(1, Objective=square(x1), Constraints=[x1 >= 10])
        x2 = Variable(name='x')
        gvx.AddNode(2, Objective=square(x2), Constraints=[x2 >= 5])

        #set rho update function to return a constant rho at every iteration
        SetRhoUpdateFunc(Func = lambda rho, res_p, thr_p, res_d, thr_d: rho)
        
        #solve the optimisation problem
        gvx.Solve(UseADMM=True)

        #check if the final solution is within 2 decimal places of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), 10, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), 5, places=2)
    
    def test_rho_exp_decay(self):
        """ Test whether SnapVX works when rho decays exponentially
        """

        #generate an empty graph
        gvx = TGraphVX()

        #add two nodes to the graph with each node having a single variable with a single constraint
        x1 = Variable(name='x')
        gvx.AddNode(1, Objective=square(x1), Constraints=[x1 >= 10])
        x2 = Variable(name='x')
        gvx.AddNode(2, Objective=square(x2), Constraints=[x2 >= 5])

        #set rho update function to return a constant rho at every iteration
        SetRhoUpdateFunc(Func = lambda rho, res_p, thr_p, res_d, thr_d: 0.999 * rho)
        
        #solve the optimisation problem
        gvx.Solve(UseADMM=True)

        #check if the final solution is within 2 decimal places of the actual solution
        self.assertAlmostEqual(gvx.GetNodeValue(1, 'x'), 10, places=2)
        self.assertAlmostEqual(gvx.GetNodeValue(2, 'x'), 5, places=2)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SetRhoUpdateTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
