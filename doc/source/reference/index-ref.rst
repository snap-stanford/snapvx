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

        Adds a node with id *NId* (:class:`int`) and the given CVXPY *Objective*
        and *Constraints* (:class:`List`).

     .. describe:: SetNodeObjective(NId, Objective)

        Sets the CVXPY Objective of node with id *NId* (:class:`int`) to be
        *Objective*.

     .. describe:: GetNodeObjective(NId)

        Returns the CVXPY Objective of the node with id *NId* (:class:`int`).

     .. describe:: SetNodeConstraints(NId, Constraints)

        Sets the CVXPY constraints of node with id *NId* (:class:`int`) to
        be *Constraints* (:class:`List`).

     .. describe:: GetNodeConstraints(NId)

        Returns the CVXPY constraints (:class:`List`) of the node with id *NId*
        (:class:`int`).

     .. describe:: AddEdge(SrcNId, DstNId, ObjectiveFunc=None, Objective=norm(0), Constraints=[])

        Adds an undirected edge {*SrcNId*, *DstNId*} (:class:`int`,
        :class:`int`). *ObjectiveFunc* is a function that allows the user to
        use the CVXPY Variables at each node endpoint without needing to
        maintain reference. It should accept two arguments, which will be a
        :class:`Dictionary` for the source and destination nodes, respectively.
        The dictionaries are of the form {:class:`string` varName : CVXPY
        Variable}. *ObjectiveFunc* should return a tuple containing (CVXPY
        Objective, CVXPY constraints (:class:`List`)). If *ObjectiveFunc* is
        *None*, then the CVXPY Objective and constraints (:class:`List`)
        parameters are used.

     .. describe:: SetEdgeObjective(SrcNId, DstNId, Objective)

        Set the CVXPY Objective of the edge {*SrcNId*, *DstNId*} (:class:`int`,
        :class:`int`) to be *Objective*.

     .. describe:: GetEdgeObjective(SrcNId, DstNId)

        Returns the CVXPY Objective of the edge {*SrcNId*, *DstNId*}
        (:class:`int`, :class:`int`).

     .. describe:: SetEdgeConstraints(SrcNId, DstNId, Constraints)

        Set the CVXPY constraints of the edge {*SrcNId*, *DstNId*}
        (:class:`int`, :class:`int`) to be *Constraints* (:class:`List`).

     .. describe:: GetEdgeConstraints(SrcNId, DstNId)

        Returns the CVXPY constraints (:class:`List`) of the of the edge
        {*SrcNId*, *DstNId*} (:class:`int`, :class:`int`).

     .. describe:: Nodes()

        Returns a generator for the nodes in the graph.

     .. describe:: Edges()

        Returns a generator for the edges in the graph.

   The following methods allow bulk loading of nodes and edges:

     .. describe:: AddNodeObjectives(Filename, ObjFunc, NodeIDs=None, IdCol=None)

        Bulk loads CVXPY Objectives for nodes, using the data in the CSV file
        with name *Filename* (:class:`string`). The file will be parsed line by
        line, and *ObjFunc* will be called once per line. *ObjFunc* should
        accept one argument, which will be a :class:`List[string]`
        containing data from that particular line. *ObjFunc* should return a
        tuple containing (CVXPY Objective, CVXPY constraints (:class:`List`))
        for that particular node. If *NodeIDs* (:class:`List[int]`) is
        specified, *ObjFunc* will be called for the nodes with ids matching
        those in the list. Otherwise, if *IdCol* (:class:`int`) is specified,
        then *ObjFunc* will be called on the node with the id matching the
        data value at that column in the CSV line. If both *NodeIDs* and *IdCol*
        are *None*, then *ObjFunc* will be called for nodes with ids in
        increasing order.

     .. describe:: AddEdgeObjectives(ObjFunc, Filename=None, EdgeIDs=None, SrcIdCol=None, DstIdCol=None)

        Bulk loads CVXPY Objectives for edges. *ObjFunc* is a function that
        allows the user to use the CVXPY Variables at each node endpoint without
        needing to maintain reference. It should accept three arguments. The
        first two arguments will :class:`Dictionary` for the source and
        destination nodes, respectively. The dictionaries are of the form
        {:class:`string` varName : CVXPY Variable}. The third argument is valid
        if a given CSV file with name *Filename* (:class:`string`) is specified.
        If so, the third argument will be a :class:`List[string]` containing
        data from that particular line. Otherwise, it will be *None*.
        *ObjFunc* should return a tuple containing (CVXPY Objective, CVXPY
        constraints (:class:`List`)). If *EdgeIDs* (:class:`List[(int, int)]`)
        is specified, *ObjFunc* will be called for the edge with ids matching
        those in the list. Otherwise, if *SrcIdCol* (:class:`int`) and
        *DstIdCol* (:class:`int`) are specified, then *ObjFunc* will be called
        on the edge with endpoints matching the data values at those columns in
        the CSV line. If *EdgeIDs*, *SrcIdCol*, and *DstIdCol* are *None*, then
        *ObjFunc* will called for edges with ids in increasing order.

   The following methods solve the optimization problem represented by the
   :class:`TGraphVX` and offer various ways to extract the solution:

     .. describe:: Solve(M=Minimize, UseADMM=True, NumProcessors=0, Rho=1.0, MaxIters=250, EpsAbs=0.01, EpsRel=0.01, Verbose=False, UseClustering=False, clusterSize = 1000)

        Adds CVXPY Objectives and constraints over all nodes and edges to form
        one collective CVXPY Problem and solves it. *M* can be the CVXPY
        function *Maximize* or *Minimize*. *UseADMM* (:class:`bool`) specifies
        whether the backend algorithm should use ADMM or one serial solver.
	*UseClustering* specifies whether the problem is to be solved for a 
	supergraph with each node being a cluster of nodes in the original graph.
	*clusterSize* specifies the maximum variable size that can be present in 
	the supernode of the supergraph. *Verbose* (:class:`bool`) can be specified 
	for verbose output. The rest of the parameters are relevant only is ADMM is used.
        *NumProcessors* specifies how many threads should be used in parallel.
        If *NumProcessors* is 0, then the number of CPUs is used as a default.
        *Rho*, *EpsAbs*, and *EpsRel* (:class:`float`) are all parameters used
        in the calculation of the convergence criteria. *EpsAbs* and *EpsRel*
        are the primal and dual thresholds, respectively. *MaxIters*
        (:class:`int`) is the maximum number of iterations for ADMM.

     .. describe:: PrintSolution(Filename=None)

        After *Solve* is called, prints the solution to the collective CVXPY
        Problem, organized by node. Prints to the file with name *Filename*
        (:class:`string`), if specified. Otherwise, prints to the console.

     .. describe:: GetNodeValue(NId, Name)

        After *Solve* is called, gets the value of the CVXPY Variable with
        name *Name* (:class:`string`) at node with id *NId* (:class:`int`).

     .. describe:: GetNodeVariables(NId)

        After *Solve* is called, returns a dictionary of all CVXPY Variables
        at the node with id *NId* (:class:`int`). The dictionary is of the form
        {:class:`string` name : CVXPY Variable}.

Static Functions
====

     .. describe:: snapvx.LoadEdgeList(Filename)

        Initializes a :class:`TGraphVX` based off the data given in the file
        with name *Filename* (:class:`string`). There should be one edge
        specified per line, written as "srcID dstID". Commented lines that begin
        with '#' are ignored.

     .. describe:: snapvx.SetRho(Rho=None)

        Updates the value of rho used in the convergence criteria. If *Rho*
        (:class:`float`) is *None*, then the default rho value of 1.0 is used.

     .. describe:: snapvx.SetRhoUpdateFunc(Func=None)

        Allows for the user to dynamically update the rho value at the end of
        each ADMM iteration. The function *Func* should accept five arguments.
        The first argument if the old value of rho. The next two arguments are
        the primal residual and threshold values from calculating the
        convergence criteria in that iteration. The last two arguments are the
        dual residual and threshold value. *Func* should return the new value of
        rho. If *Func* is *None*, then the default behavior of not updating
        rho is set.
