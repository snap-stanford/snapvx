SnapVX Reference Manual
------------------------

This document is a reference guide to SnapVX functionality.

SnapVX is a python-based convex optimization solver for problems defined on
graphs. For problems of this form, SnapVX provides a fast and scalable solution
with guaranteed global convergence. It combines the graph capabilities of
Snap.py with the convex solver from CVXPY.

To use SnapVX in Python, import the **snapvx** module:

>>> import snapvx

The code in this document assumes that SnapVX has been imported as shown above.

The core functionality of SnapVX lies in the :class:`TGraphVX` class, a subclass
of :class:`TUNGraph` from Snap.py. :class:`TGraphVX` has extended functionality
to incorporate CVXPY elements.

TGraphVX
====

.. class:: TGraphVX()
           TGraphVX(Graph)

   Returns a new :class:`TGraphVX` initialized with the same nodes and edges
   specified by optional parameter *Graph*. If no *Graph* is given, then the
   :class:`TGraphVX` starts an empty graph.

   In addition to all of the methods of :class:`TUNGraph` from Snapy.py,
   below is a list of new methods supported by the :class:`TGraphVX` class:

   The following basic methods assist in constructing a :class:`TGraphVX`
   while incorporating CVXPY elements:

     .. describe:: AddNode(NId, Objective=norm(0), Constraints=[])

        Adds a node with id *NId* (an :class:`int`) and the given CVXPY
        *Objective* and *Constraints*.

     .. describe:: SetNodeObjective(NId, Objective)

        Sets the CVXPY Objective of node with id *NId* to be *Objective*.

     .. describe:: GetNodeObjective(NId)

        Returns the CVXPY Objective of the node with id *NId*.

     .. describe:: SetNodeConstraints(NId, Constraints)

        Sets the CVXPY constraints (a :class:`List`) of node with id *NId* to
        be *Constraints*.

     .. describe:: GetNodeConstraints(NId)

        Returns the CVXPY constraints of the node with id *NId*.

     .. describe:: AddEdge(SrcNId, DstNId, ObjectiveFunc=None, Objective=norm(0), Constraints=[])

        TODO

     .. describe:: SetEdgeObjective(SrcNId, DstNId, Objective)

        TODO

     .. describe:: GetEdgeObjective(SrcNId, DstNId)

        TODO

     .. describe:: SetEdgeConstraints(SrcNId, DstNId, Constraints)

        TODO

     .. describe:: GetEdgeConstraints(SrcNId, DstNId)

        TODO

     .. describe:: Nodes()

        TODO

     .. describe:: Edges()

        TODO

   The following methods allow bulk loading of nodes and edges:

     .. describe:: AddNodeObjectives(Filename, ObjFunc, NodeIDs=None, IdCol=None)

        TODO

     .. describe:: AddEdgeObjectives(ObjFunc, Filename=None, EdgeIDs=None, SrcIdCol=None, DstIdCol=None)

        TODO

   The following methods solve the optimization problem represented by the
   :class:`TGraphVX` and offer various ways to extract the solution:

     .. describe:: Solve(M=Minimize, UseADMM=True, NumProcessors=0, Rho=1.0, MaxIters=250, EpsAbs=0.01, EpsRel=0.01, Verbose=False)

        TODO

     .. describe:: PrintSolution(Filename=None)

        TODO

     .. describe:: GetNodeValue(NId, Name)

        TODO

     .. describe:: GetNodeVariables(NId)

        TODO

Static Functions
====

     .. describe:: snapvx.LoadEdgeList(Filename)

        TODO

     .. describe:: snapvx.SetRho(Rho=None)

        TODO

     .. describe:: snapvx.SetRhoUpdateFunc(Func=None)

        TODO
