from snapvx import *

#Create new graph
gvx = TGraphVX()

#Use CVXPY syntax to define a problem
x1 = Variable(1, name='x1')
obj = square(x1)
#Add Node 1 with the given objective, with the constraint that x1 <= 10
gvx.AddNode(1, Objective=obj, Constraints=[x1 <= 10])

#Similarly, add Node 2 with objective |x2 + 3|
x2 = Variable(1, name='x2')
obj2 = abs(x2 + 3)
gvx.AddNode(2, obj2, [])

#Add an edge between the two nodes, 
#Define an objective, constraints using CVXPY syntax
gvx.AddEdge(1, 2, Objective=square(norm(x1 - x2)), Constraints=[])

#Solve the problem, and print the solution
gvx.Solve()
print gvx.PrintSolution()
