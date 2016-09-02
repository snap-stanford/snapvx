import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import numpy as np
from snapvx import *
from cvxpy import *
import unittest

class NoSharedVarsTest(BaseTest):

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
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(NoSharedVarsTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
