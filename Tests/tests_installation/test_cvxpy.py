import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import unittest
import numpy as np
from cvxpy import *

""" Test that CVXPY installed correctly.
"""
class CVXPYTest(BaseTest):

    def test_cvxpy(self):
        """ Test that CVXPY installed correctly.
        """
        ## Test taken from test_advanded() [sic] from cvxpy/tests/test_examples.py
        ## of the CVXPY repository.

        # Solving a problem with different solvers.
        x = Variable(2)
        obj = Minimize(x[0] + norm(x, 1))
        constraints = [x >= 2]
        prob = Problem(obj, constraints)

        # Solve with ECOS.
        prob.solve(solver=ECOS)
        print("optimal value with ECOS:", prob.value)
        self.assertAlmostEqual(prob.value, 6)

        # Solve with ECOS_BB.
        prob.solve(solver=ECOS_BB)
        print("optimal value with ECOS_BB:", prob.value)
        self.assertAlmostEqual(prob.value, 6)

        # Solve with CVXOPT.
        prob.solve(solver=CVXOPT)
        print("optimal value with CVXOPT:", prob.value)
        self.assertAlmostEqual(prob.value, 6)

        # Solve with SCS.
        prob.solve(solver=SCS)
        print("optimal value with SCS:", prob.value)
        self.assertAlmostEqual(prob.value, 6, places=2)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(CVXPYTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
