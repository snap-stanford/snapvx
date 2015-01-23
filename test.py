## snapvx Tests

from snapvx import *

# Simple test on a graph with 2 nodes and 1 edge.
# Node 1: objective = x1^2
# Node 2: objective = |x2 + 3|
# Edge: objective = ||x1 - x2||^2
# No constraints.
# Expect optimal status with a value of 2.5
# x1 = -0.5, x2 = -1
def test1():
    gvx = TUNGraphVX(2, 1)
    x1 = Variable()
    x2 = Variable()
    objNode1 = square(x1)
    objNode2 = abs(x2 + 3)
    gvx.AddNode(1, objNode1, x1)
    gvx.AddNode(2, objNode2, x2)

    # Test getting Variables via gvx
    n1Var = gvx.GetNodeVariable(1)
    n2Var = gvx.GetNodeVariable(2)
    objEdge = square(norm(n1Var - n2Var))
    gvx.AddEdge(1, 2, objEdge)

    gvx.Solve()
    print gvx.status, gvx.value
    print x1.value, x2.value


def main():
    print '** Starting Test 1 **'
    test1()
    print '** Done **'

if __name__ == "__main__":
    main()
