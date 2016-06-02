from snapvx import *
import numpy as np
import numpy.linalg as alg
import scipy as spy

import time
from itertools import *
import sys

#solve the network lasso problem consisting of a central node, and a dummy node.
#We maximise the log of the determinant of a semidefinite matrix
numFeatures = 2
lamb = 10

np.random.seed(1)
gvx = TGraphVX()
empCov = np.random.randn(numFeatures+1,100*(numFeatures+1))
empCov = np.cov(empCov)
S = semidefinite(numFeatures+1,name='S')

#add the two nodes
gvx.AddNode(0,Objective=trace(S*empCov)-log_det(S))
gvx.AddNode(1)

#add an edge between them with the regularisation penalty
gvx.AddEdge(0, 1,Objective=lamb*norm(S,1))

#Solve the problem
gvx.Solve(Verbose=True,Rho=1.0,UseADMM=True,MaxIters=20)
gvx.PrintSolution()
