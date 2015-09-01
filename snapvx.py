## snapvx

from snap import *
from cvxpy import *

import math
import multiprocessing
import numpy
from scipy.sparse import lil_matrix
import sys
import time


# File format: One edge per line, written as "srcID dstID"
# Commented lines that start with '#' are ignored
# Returns a TGraphVX object with the designated edges and nodes
def LoadEdgeList(filename):
    gvx = TGraphVX()
    nids = set()
    infile = open(filename, 'r')
    for line in infile:
        if line.startswith('#'): continue
        [src, dst] = line.split()
        if int(src) not in nids:
            gvx.AddNode(int(src))
            nids.add(int(src))
        if int(dst) not in nids:
            gvx.AddNode(int(dst))
            nids.add(int(dst))
        gvx.AddEdge(int(src), int(dst))
    return gvx


# TGraphVX inherits from the TUNGraph object defined by Snap.py
class TGraphVX(TUNGraph):

    __default_objective = norm(0)
    __default_constraints = []

    # Data Structures
    # ---------------
    # node_objectives  = {int NId : CVXPY Expression}
    # node_constraints = {int NId : [CVXPY Constraint]}
    # edge_objectives  = {(int NId1, int NId2) : CVXPY Expression}
    # edge_constraints = {(int NId1, int NId2) : [CVXPY Constraint]}
    #
    # ADMM-Specific Structures
    # ------------------------
    # node_variables   = {int NId :
    #       [(CVXPY Variable id, CVXPY Variable name, CVXPY Variable, offset)]}
    # node_values = {int NId : numpy array}
    # node_values points to the numpy array containing the value of the entire
    #     variable space corresponding to then node. Use the offset to get the
    #     value for a specific variable.
    #
    # Constructor
    # If Graph is a Snap.py graph, initializes a SnapVX graph with the same
    # nodes and edges.
    def __init__(self, Graph=None):
        # Initialize data structures
        self.node_objectives = {}
        self.node_variables = {}
        self.node_constraints = {}
        self.edge_objectives = {}
        self.edge_constraints = {}
        self.node_values = {}
        self.status = None
        self.value = None

        # Initialize superclass
        nodes = 0
        edges = 0
        if Graph != None:
            nodes = Graph.GetNodes()
            edges = Graph.GetEdges()
        TUNGraph.__init__(self, nodes, edges)

        # Support for constructor with Snap.py graph argument
        if Graph != None:
            for ni in Graph.Nodes():
                self.AddNode(ni.GetId())
            for ei in Graph.Edges():
                self.AddEdge(ei.GetSrcNId(), ei.GetDstNId())

    # Simple iterator to iterator over all nodes in graph. Similar in
    # functionality to Nodes() iterator of PUNGraph in Snap.py.
    def Nodes(self):
        ni = TUNGraph.BegNI(self)
        for i in xrange(TUNGraph.GetNodes(self)):
            yield ni
            ni.Next()

    # Simple iterator to iterator over all edge in graph. Similar in
    # functionality to Edges() iterator of PUNGraph in Snap.py.
    def Edges(self):
        ei = TUNGraph.BegEI(self)
        for i in xrange(TUNGraph.GetEdges(self)):
            yield ei
            ei.Next()

    # Adds objectives together to form one collective CVXPY Problem.
    # Option of specifying Maximize() or the default Minimize() (only works when useADMM=False).
    # Graph status and value properties will also be set.
    # Individual variable values can be retrieved using GetNodeValue().
    # Option to use serial version or distributed ADMM.
    # maxIters optional parameter: Maximum iterations for distributed ADMM.
    def Solve(self, M=Minimize, useADMM=True, rho=1.0, maxIters=250,
              verbose=False):
        if useADMM:
            self.__SolveADMM(rho, maxIters, verbose)
            return
        if verbose:
            print 'Serial ADMM'
        objective = 0
        constraints = []
        # Add all node objectives and constraints
        for ni in self.Nodes():
            nid = ni.GetId()
            objective += self.node_objectives[nid]
            constraints += self.node_constraints[nid]
        # Add all edge objectives and constraints
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            objective += self.edge_objectives[etup]
            constraints += self.edge_constraints[etup]
        # Solve CVXPY Problem
        objective = M(objective)
        problem = Problem(objective, constraints)
        problem.solve()
        if problem.status in [INFEASIBLE_INACCURATE, UNBOUNDED_INACCURATE]:
            problem.solve(solver=SCS)
        # Set TGraphVX status and value to match CVXPY
        self.status = problem.status
        self.value = problem.value
        # Insert into hash to support ADMM structures and GetNodeValue()
        for ni in self.Nodes():
            nid = ni.GetId()
            variables = self.node_variables[nid]
            value = None
            for (varID, varName, var, offset) in variables:
                if var.size[0] == 1:
                    val = numpy.array([var.value])
                else:
                    val = numpy.array(var.value).reshape(-1,)
                if not value:
                    value = val
                else:
                    value = numpy.concatenate((value, val))
            self.node_values[nid] = value

    # Implementation of distributed ADMM
    # Uses a global value of rho_param for rho
    # Will run for a maximum of maxIters iterations
    def __SolveADMM(self, rho_param, maxIters, verbose=False):
        global node_vals, edge_z_vals, edge_u_vals, rho
        global getValue, rho_update_func

        num_processors = multiprocessing.cpu_count()
        rho = rho_param
        if verbose:
            print 'Distributed ADMM (%d processors)' % num_processors

        # Organize information for each node in helper node_info structure
        node_info = {}
        # Keeps track of the current offset necessary into the shared node
        # values Array
        length = 0
        for ni in self.Nodes():
            nid = ni.GetId()
            deg = ni.GetDeg()
            obj = self.node_objectives[nid]
            variables = self.node_variables[nid]
            con = self.node_constraints[nid]
            neighbors = [ni.GetNbrNId(j) for j in xrange(deg)]
            # Node's constraints include those imposed by edges
            for neighborId in neighbors:
                etup = self.__GetEdgeTup(nid, neighborId)
                econ = self.edge_constraints[etup]
                con += econ
            # Calculate sum of dimensions of all Variables for this node
            size = 0
            for (varID, varName, var, offset) in variables:
                size += var.size[0]
            # Nearly complete information package for this node
            node_info[nid] = (nid, obj, variables, con, length, size, deg,\
                neighbors)
            length += size
        node_vals = multiprocessing.Array('d', [0.0] * length)
        x_length = length

        # Organize information for each node in final edge_list structure and
        # also helper edge_info structure
        edge_list = []
        edge_info = {}
        # Keeps track of the current offset necessary into the shared edge
        # values Arrays
        length = 0
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            obj = self.edge_objectives[etup]
            con = self.edge_constraints[etup]
            con += self.node_constraints[etup[0]] +\
                self.node_constraints[etup[1]]
            # Get information for each endpoint node
            info_i = node_info[etup[0]]
            info_j = node_info[etup[1]]
            ind_zij = length
            ind_uij = length
            length += info_i[X_LEN]
            ind_zji = length
            ind_uji = length
            length += info_j[X_LEN]
            # Information package for this edge
            tup = (etup, obj, con,\
                info_i[X_VARS], info_i[X_LEN], info_i[X_IND], ind_zij, ind_uij,\
                info_j[X_VARS], info_j[X_LEN], info_j[X_IND], ind_zji, ind_uji)
            edge_list.append(tup)
            edge_info[etup] = tup
        edge_z_vals = multiprocessing.Array('d', [0.0] * length)
        edge_u_vals = multiprocessing.Array('d', [0.0] * length)
        z_length = length

        # Populate sparse matrix A.
        # A has dimensions (p, n), where p is the length of the stacked vector
        # of node variables, and n is the length of the stacked z vector of
        # edge variables.
        # Each row of A has one 1. There is a 1 at (i,j) if z_i = x_j.
        A = lil_matrix((z_length, x_length), dtype=numpy.int8)
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            info_edge = edge_info[etup]
            info_i = node_info[etup[0]]
            info_j = node_info[etup[1]]
            for offset in xrange(info_i[X_LEN]):
                row = info_edge[Z_ZIJIND] + offset
                col = info_i[X_IND] + offset
                A[row, col] = 1
            for offset in xrange(info_j[X_LEN]):
                row = info_edge[Z_ZJIIND] + offset
                col = info_j[X_IND] + offset
                A[row, col] = 1
        A_tr = A.transpose()

        # Create final node_list structure by adding on information for
        # node neighbors
        node_list = []
        for nid, info in node_info.iteritems():
            entry = [nid, info[X_OBJ], info[X_VARS], info[X_CON], info[X_IND],\
                info[X_LEN], info[X_DEG]]
            # Append information about z- and u-value indices for each
            # node neighbor
            for i in xrange(info[X_DEG]):
                neighborId = info[X_NEIGHBORS][i]
                indices = (Z_ZIJIND, Z_UIJIND) if nid < neighborId else\
                    (Z_ZJIIND, Z_UJIIND)
                einfo = edge_info[self.__GetEdgeTup(nid, neighborId)]
                entry.append(einfo[indices[0]])
                entry.append(einfo[indices[1]])
            node_list.append(entry)

        pool = multiprocessing.Pool(num_processors)
        num_iterations = 0
        z_old = getValue(edge_z_vals, 0, z_length)
        # Proceed until convergence criteria are achieved or the maximum
        # number of iterations has passed
        while num_iterations <= maxIters:
            # Check convergence criteria
            if num_iterations != 0:
                x = getValue(node_vals, 0, x_length)
                z = getValue(edge_z_vals, 0, z_length)
                u = getValue(edge_u_vals, 0, z_length)
                # Determine if algorithm should stop. Retrieve primal and dual
                # residuals and thresholds
                stop, res_pri, e_pri, res_dual, e_dual =\
                    self.__CheckConvergence(A, A_tr, x, z, z_old, u, rho,\
                                            x_length, z_length, verbose)
                if stop: break
                z_old = z
                # Update rho and scale u-values
                rho_new = rho_update_func(rho, res_pri, e_pri, res_dual, e_dual)
                scale = float(rho) / rho_new
                edge_u_vals[:] = [i * scale for i in edge_u_vals]
                rho = rho_new
            num_iterations += 1

            if verbose:
                # Debugging information prints current iteration #
                print 'Iteration %d' % num_iterations
            pool.map(ADMM_x, node_list)
            pool.map(ADMM_z, edge_list)
            pool.map(ADMM_u, edge_list)
        pool.close()
        pool.join()

        # Insert into hash to support GetNodeValue()
        for entry in node_list:
            nid = entry[X_NID]
            index = entry[X_IND]
            size = entry[X_LEN]
            self.node_values[nid] = getValue(node_vals, index, size)
        # Set TGraphVX status and value to match CVXPY
        self.status = 'Optimal' if num_iterations <= maxIters else 'Incomplete: max iterations reached'
        self.value = self.GetTotalProblemValue()

    # Iterate through all variables and update values.
    # Sum all objective values over all nodes and edges.
    def GetTotalProblemValue(self):
        global getValue
        result = 0.0
        for ni in self.Nodes():
            nid = ni.GetId()
            for (varID, varName, var, offset) in self.node_variables[nid]:
                var.value = self.GetNodeValue(nid, varName)
        for ni in self.Nodes():
            result += self.node_objectives[ni.GetId()].value
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            result += self.edge_objectives[etup].value
        return result

    # Returns True if convergence criteria have been satisfied
    # eps_abs = eps_rel = 0.01
    # r = Ax - z
    # s = rho * (A^T)(z - z_old)
    # e_pri = sqrt(p) * e_abs + e_rel * max(||Ax||, ||z||)
    # e_dual = sqrt(n) * e_abs + e_rel * ||rho * (A^T)u||
    # Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
    # Returns (boolean shouldStop, primal residual value, primal threshold,
    #          dual residual value, dual threshold)
    def __CheckConvergence(self, A, A_tr, x, z, z_old, u, rho, p, n, verbose):
        norm = numpy.linalg.norm
        e_abs = 0.01
        e_rel = 0.01
        Ax = A.dot(x)
        r = Ax - z
        s = rho * A_tr.dot(z - z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = math.sqrt(p) * e_abs + e_rel * max(norm(Ax), norm(z)) + .0001
        e_dual = math.sqrt(n) * e_abs + e_rel * norm(rho * A_tr.dot(u)) + .0001
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        if verbose:
            # Debugging information to print convergence criteria values
            print '  r:', res_pri
            print '  e_pri:', e_pri
            print '  s:', res_dual
            print '  e_dual:', e_dual
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return (stop, res_pri, e_pri, res_dual, e_dual)

    # API to get node Variable value after solving with ADMM.
    def GetNodeValue(self, NId, name):
        self.__VerifyNId(NId)
        for (varID, varName, var, offset) in self.node_variables[NId]:
            if varName == name:
                offset = offset
                value = self.node_values[NId]
                return value[offset:(offset + var.size[0])]
        return None

    # Prints value of all node variables to console or file, if given
    def PrintSolution(self, filename=None):
        numpy.set_printoptions(linewidth=numpy.inf)
        out = sys.stdout if (filename == None) else open(filename, 'w+')

        out.write('Status: %s\n' % self.status)
        out.write('Total Objective: %f\n' % self.value)
        for ni in self.Nodes():
            nid = ni.GetId()
            s = 'Node %d:\n' % nid
            out.write(s)
            for (varID, varName, var, offset) in self.node_variables[nid]:
                val = numpy.transpose(self.GetNodeValue(nid, varName))
                s = '  %s %s\n' % (varName, str(val))
                out.write(s)

    # Helper method to verify existence of an NId.
    def __VerifyNId(self, NId):
        if not TUNGraph.IsNode(self, NId):
            raise Exception('Node %d does not exist.' % NId)

    # Helper method to get CVXPY Variables out of a CVXPY Objective
    def __ExtractVariableList(self, Objective):
        l = [(var.name(), var) for var in Objective.variables()]
        # Sort in ascending order by name
        l.sort(key=lambda t: t[0])
        l2 = []
        offset = 0
        for (varName, var) in l:
            # Add tuples of the form (id, name, object, offset)
            l2.append((var.id, varName, var, offset))
            offset += var.size[0]
        return l2

    # Adds a Node to the TUNGraph and stores the corresponding CVX information.
    def AddNode(self, NId, Objective=__default_objective,\
            Constraints=__default_constraints):
        self.node_objectives[NId] = Objective
        self.node_variables[NId] = self.__ExtractVariableList(Objective)
        self.node_constraints[NId] = Constraints
        return TUNGraph.AddNode(self, NId)

    def SetNodeObjective(self, NId, Objective):
        self.__VerifyNId(NId)
        self.node_objectives[NId] = Objective
        self.node_variables[NId] = self.__ExtractVariableList(Objective)

    def GetNodeObjective(self, NId):
        self.__VerifyNId(NId)
        return self.node_objectives[NId]

    def SetNodeConstraints(self, NId, Constraints):
        self.__VerifyNId(NId)
        self.node_constraints[NId] = Constraints

    def GetNodeConstraints(self, NId):
        self.__VerifyNId(NId)
        return self.node_constraints[NId]

    # Helper method to get a tuple representing an edge. The smaller NId
    # goes first.
    def __GetEdgeTup(self, NId1, NId2):
        return (NId1, NId2) if NId1 < NId2 else (NId2, NId1)

    # Helper method to verify existence of an edge.
    def __VerifyEdgeTup(self, ETup):
        if not TUNGraph.IsEdge(self, ETup[0], ETup[1]):
            raise Exception('Edge {%d,%d} does not exist.' % ETup)

    # Adds an Edge to the TUNGraph and stores the corresponding CVX information.
    # obj_func is a function which accepts two arguments, a dictionary of
    #     variables for the source and destination nodes
    #     { string varName : CVXPY Variable }
    # obj_func should return a tuple of (objective, constraints), although
    #     it will assume a singleton object will be an objective and will use
    #     the default constraints.
    # If obj_func is None, then will use Objective and Constraints, which are
    #     parameters currently set to defaults.
    def AddEdge(self, SrcNId, DstNId, Objective_Func=None,
            Objective=__default_objective, Constraints=__default_constraints):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        if Objective_Func != None:
            src_vars = self.GetNodeVariables(SrcNId)
            dst_vars = self.GetNodeVariables(DstNId)
            ret = Objective_Func(src_vars, dst_vars)
            if type(ret) is tuple:
                # Tuple = assume we have (objective, constraints)
                self.edge_objectives[ETup] = ret[0]
                self.edge_constraints[ETup] = ret[1]
            else:
                # Singleton object = assume it is the objective
                self.edge_objectives[ETup] = ret
                self.edge_constraints[ETup] = self.__default_constraints
        else:
            self.edge_objectives[ETup] = Objective
            self.edge_constraints[ETup] = Constraints
        return TUNGraph.AddEdge(self, SrcNId, DstNId)

    def SetEdgeObjective(self, SrcNId, DstNId, Objective=__default_objective):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        self.edge_objectives[ETup] = Objective

    def GetEdgeObjective(self, SrcNId, DstNId):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        return self.edge_objectives[ETup]

    def SetEdgeConstraints(self, SrcNId, DstNId, Constraints):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        self.edge_constraints[ETup] = Constraints

    def GetEdgeConstraints(self, SrcNId, DstNId):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        return self.edge_constraints[ETup]


    # Returns a dictionary of all variables corresponding to a node.
    # { string name : CVXPY Variable }
    # This can be used in place of bulk loading functions to recover necessary
    # Variables for an edge.
    def GetNodeVariables(self, NId):
        self.__VerifyNId(NId)
        d = {}
        for (varID, varName, var, offset) in self.node_variables[NId]:
            d[varName] = var
        return d

    # Bulk loading for nodes
    # obj_func is a function which accepts one argument, an array of strings
    #     parsed from the given CSV filename
    # obj_func should return a tuple of (objective, constraints), although
    #     it will assume a singleton object will be an objective
    # Optional parameter nodeIDs allows the user to pass in a list specifying,
    # in order, the node IDs that correspond to successive rows
    # If nodeIDs is None, then the file must have a column denoting the
    # node ID for each row. The index of this column (0-indexed) is idCol.
    # If nodeIDs and idCol are both None, then will iterate over all Nodes, in
    # order, as long as the file lasts
    def AddNodeObjectives(self, filename, obj_func, nodeIDs=None, idCol=None):
        infile = open(filename, 'r')
        if nodeIDs == None and idCol == None:
            stop = False
            for ni in self.Nodes():
                nid = ni.GetId()
                while True:
                    line = infile.readline()
                    if line == '': stop = True
                    if not line.startswith('#'): break
                if stop: break
                data = [x.strip() for x in line.split(',')]
                ret = obj_func(data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetNodeObjective(nid, ret[0])
                    self.SetNodeConstraints(nid, ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetNodeObjective(nid, ret)
        if nodeIDs == None:
            for line in infile:
                if line.startswith('#'): continue
                data = [x.strip() for x in line.split(',')]
                ret = obj_func(data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetNodeObjective(int(data[idCol]), ret[0])
                    self.SetNodeConstraints(int(data[idCol]), ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetNodeObjective(int(data[idCol]), ret)
        else:
            for nid in nodeIDs:
                while True:
                    line = infile.readline()
                    if line == '':
                        raise Exception('File %s is too short.' % filename)
                    if not line.startswith('#'): break
                data = [x.strip() for x in line.split(',')]
                ret = obj_func(data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetNodeObjective(nid, ret[0])
                    self.SetNodeConstraints(nid, ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetNodeObjective(nid, ret)
        infile.close()

    # Bulk loading for edges
    # If filename is None:
    # obj_func is a function which accepts three arguments, a dictionary of
    #     variables for the source and destination nodes, and an unused param
    #     { string varName : CVXPY Variable } x2, None
    # obj_func should return a tuple of (objective, constraints), although
    #     it will assume a singleton object will be an objective
    # If filename exists:
    # obj_func is the same, except the third param will be be an array of
    #     strings parsed from the given CSV filename
    # Optional parameter edgeIDs allows the user to pass in a list specifying,
    # in order, the edgeIDs that correspond to successive rows. An edgeID is
    # a tuple of (srcID, dstID).
    # If edgeIDs is None, then the file may have columns denoting the srcID and
    # dstID for each row. The indices of these columns are 0-indexed.
    # If edgeIDs and id columns are None, then will iterate through all edges
    # in order, as long as the file lasts.
    def AddEdgeObjectives(self, obj_func, filename=None, edgeIDs=None,\
            srcIdCol=None, dstIdCol=None):
        if filename == None:
            for ei in self.Edges():
                src_id = ei.GetSrcNId()
                src_vars = self.GetNodeVariables(src_id)
                dst_id = ei.GetDstNId()
                dst_vars = self.GetNodeVariables(dst_id)
                ret = obj_func(src_vars, dst_vars, None)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetEdgeObjective(src_id, dst_id, ret[0])
                    self.SetEdgeConstraints(src_id, dst_id, ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetEdgeObjective(src_id, dst_id, ret)
            return
        infile = open(filename, 'r')
        if edgeIDs == None and (srcIdCol == None or dstIdCol == None):
            stop = False
            for ei in self.Edges():
                src_id = ei.GetSrcNId()
                src_vars = self.GetNodeVariables(src_id)
                dst_id = ei.GetDstNId()
                dst_vars = self.GetNodeVariables(dst_id)
                while True:
                    line = infile.readline()
                    if line == '': stop = True
                    if not line.startswith('#'): break
                if stop: break
                data = [x.strip() for x in line.split(',')]
                ret = obj_func(src_vars, dst_vars, data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetEdgeObjective(src_id, dst_id, ret[0])
                    self.SetEdgeConstraints(src_id, dst_id, ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetEdgeObjective(src_id, dst_id, ret)
        if edgeIDs == None:
            for line in infile:
                if line.startswith('#'): continue
                data = [x.strip() for x in line.split(',')]
                src_id = int(data[srcIdCol])
                dst_id = int(data[dstIdCol])
                src_vars = self.GetNodeVariables(src_id)
                dst_vars = self.GetNodeVariables(dst_id)
                ret = obj_func(src_vars, dst_vars, data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetEdgeObjective(src_id, dst_id, ret[0])
                    self.SetEdgeConstraints(src_id, dst_id, ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetEdgeObjective(src_id, dst_id, ret)
        else:
            for edgeID in edgeIDs:
                etup = self.__GetEdgeTup(edgeID[0], edgeID[1])
                while True:
                    line = infile.readline()
                    if line == '':
                        raise Exception('File %s is too short.' % filename)
                    if not line.startswith('#'): break
                data = [x.strip() for x in line.split(',')]
                src_vars = self.GetNodeVariables(etup[0])
                dst_vars = self.GetNodeVariables(etup[1])
                ret = obj_func(src_vars, dst_vars, data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetEdgeObjective(etup[0], etup[1], ret[0])
                    self.SetEdgeConstraints(etup[0], etup[1], ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetEdgeObjective(etup[0], etup[1], ret)
        infile.close()


## ADMM Global Variables and Functions ##

# By default, rho is 1.0. Default rho update is identity function and does not
# depend on primal or dual residuals or thresholds.
__default_rho = 1.0
__default_rho_update_func = lambda rho, res_p, thr_p, res_d, thr_d: rho
rho = __default_rho
# Rho update function takes 5 parameters
# - Old value of rho
# - Primal residual and threshold
# - Dual residual and threshold
rho_update_func = __default_rho_update_func

def SetRho(rho_new=None):
    global rho
    rho = rho_new if rho_new else __default_rho

# Rho update function should take one parameter: old_rho
# Returns new_rho
# This function will be called at the end of every iteration
def SetRhoUpdateFunc(f=None):
    global rho_update_func
    rho_update_func = f if f else __default_rho_update_func

# Tuple of indices to identify the information package for each node. Actual
# length of specific package (list) may vary depending on node degree.
# X_NID: Node ID
# X_OBJ: CVXPY Objective
# X_VARS: CVXPY Variables (entry from node_variables structure)
# X_CON: CVXPY Constraints
# X_IND: Starting index into shared node_vals Array
# X_LEN: Total length (sum of dimensions) of all variables
# X_DEG: Number of neighbors
# X_NEIGHBORS: Placeholder for information about each neighbors
#   Information for each neighbor is two entries, appended in order.
#   Starting index of the corresponding z-value in edge_z_vals. Then for u.
(X_NID, X_OBJ, X_VARS, X_CON, X_IND, X_LEN, X_DEG, X_NEIGHBORS) = range(8)

# Tuple of indices to identify the information package for each edge.
# Z_EID: Edge ID / tuple
# Z_OBJ: CVXPY Objective
# Z_CON: CVXPY Constraints
# Z_[IJ]VARS: CVXPY Variables for Node [ij] (entry from node_variables)
# Z_[IJ]LEN: Total length (sum of dimensions) of all variables for Node [ij]
# Z_X[IJ]IND: Starting index into shared node_vals Array for Node [ij]
# Z_Z[IJ|JI]IND: Starting index into shared edge_z_vals Array for edge [ij|ji]
# Z_U[IJ|JI]IND: Starting index into shared edge_u_vals Array for edge [ij|ji]
(Z_EID, Z_OBJ, Z_CON, Z_IVARS, Z_ILEN, Z_XIIND, Z_ZIJIND, Z_UIJIND,\
    Z_JVARS, Z_JLEN, Z_XJIND, Z_ZJIIND, Z_UJIIND) = range(13)

# Contain all x, z, and u values for each node and/or edge in ADMM. Use the
# given starting index and length with getValue() to get individual node values
node_vals = None
edge_z_vals = None
edge_u_vals = None

# Extract a numpy array value from a shared Array.
# Give shared array, starting index, and total length.
def getValue(arr, index, length):
    return numpy.array(arr[index:(index + length)])

# Write value of numpy array nparr (with given length) to a shared Array at
# the given starting index.
def writeValue(sharedarr, index, nparr, length):
    if length == 1:
        nparr = [nparr]
    sharedarr[index:(index + length)] = nparr

# Write the values for all of the Variables involved in a given Objective to
# the given shared Array.
# variables should be an entry from the node_values structure.
def writeObjective(sharedarr, index, objective, variables):
    for v in objective.variables():
        vID = v.id
        value = v.value
        # Find the tuple in variables with the same ID. Take the offset.
        # If no tuple exists, then silently skip.
        for (varID, varName, var, offset) in variables:
            if varID == vID:
                writeValue(sharedarr, index + offset, value, var.size[0])
                break

# x-update for ADMM for one node
def ADMM_x(entry):
    global rho
    variables = entry[X_VARS]
    norms = 0
    # Iterate through all neighbors of the node
    for i in xrange(entry[X_DEG]):
        z_index = X_NEIGHBORS + (2 * i)
        u_index = z_index + 1
        zi = entry[z_index]
        ui = entry[u_index]
        # Add norm for Variables corresponding to the node
        for (varID, varName, var, offset) in variables:
            z = getValue(edge_z_vals, zi + offset, var.size[0])
            u = getValue(edge_u_vals, ui + offset, var.size[0])
            norms += square(norm(var - z + u))

    objective = entry[X_OBJ] + (rho / 2) * norms
    objective = Minimize(objective)
    constraints = entry[X_CON]
    problem = Problem(objective, constraints)
    problem.solve()
    if problem.status in [INFEASIBLE_INACCURATE, UNBOUNDED_INACCURATE]:
        print "ECOS error: using SCS for x update"
        problem.solve(solver=SCS)

    # Write back result of x-update
    writeObjective(node_vals, entry[X_IND], objective, variables)
    return None

# z-update for ADMM for one edge
def ADMM_z(entry):
    global rho
    objective = entry[Z_OBJ]
    constraints = entry[Z_CON]
    norms = 0

    variables_i = entry[Z_IVARS]
    for (varID, varName, var, offset) in variables_i:
        x_i = getValue(node_vals, entry[Z_XIIND] + offset, var.size[0])
        u_ij = getValue(edge_u_vals, entry[Z_UIJIND] + offset, var.size[0])
        norms += square(norm(x_i - var + u_ij))

    variables_j = entry[Z_JVARS]
    for (varID, varName, var, offset) in variables_j:
        x_j = getValue(node_vals, entry[Z_XJIND] + offset, var.size[0])
        u_ji = getValue(edge_u_vals, entry[Z_UJIIND] + offset, var.size[0])
        norms += square(norm(x_j - var + u_ji))

    objective = Minimize(objective + (rho / 2) * norms)
    problem = Problem(objective, constraints)
    problem.solve()
    if problem.status in [INFEASIBLE_INACCURATE, UNBOUNDED_INACCURATE]:
        print "ECOS error: using SCS for z update"
        problem.solve(solver=SCS)

    # Write back result of z-update. Must write back for i- and j-node
    writeObjective(edge_z_vals, entry[Z_ZIJIND], objective, variables_i)
    writeObjective(edge_z_vals, entry[Z_ZJIIND], objective, variables_j)
    return None

# u-update for ADMM for one edge
def ADMM_u(entry):
    global rho
    size_i = entry[Z_ILEN]
    uij = getValue(edge_u_vals, entry[Z_UIJIND], size_i) +\
          getValue(node_vals, entry[Z_XIIND], size_i) -\
          getValue(edge_z_vals, entry[Z_ZIJIND], size_i)
    writeValue(edge_u_vals, entry[Z_UIJIND], uij, size_i)

    size_j = entry[Z_JLEN]
    uji = getValue(edge_u_vals, entry[Z_UJIIND], size_j) +\
          getValue(node_vals, entry[Z_XJIND], size_j) -\
          getValue(edge_z_vals, entry[Z_ZJIIND], size_j)
    writeValue(edge_u_vals, entry[Z_UJIIND], uji, size_j)
    return entry
