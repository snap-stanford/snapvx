## snapvx

from snap import *
from cvxpy import *
from multiprocessing import *
import numpy
import time


class TUNGraphVX(TUNGraph):

    # node_objectives  = {int NId : CVXPY Expression}
    # node_variables   = {int NId : CVXPY Variable}
    # node_constraints = {int NId : [CVXPY Constraint]}
    # edge_objectives  = {(int NId1, int NId2) : CVXPY Expression}
    # edge_constraints = {(int NId1, int NId2) : [CVXPY Constraint]}
    # (ADMM) node_values = {int NId : numpy array}
    def __init__(self, Nodes=0, Edges=0):
        self.node_objectives = {}
        self.node_variables = {}
        self.node_constraints = {}
        self.edge_objectives = {}
        self.edge_constraints = {}
        self.node_values = {}
        TUNGraph.__init__(self, Nodes, Edges)

    # Iterates through all nodes and edges. Currently adds objectives together.
    # Option of specifying Maximize() or the default Minimize().
    # Option to use ADMM.
    # Graph status and value properties will be set.
    def Solve(self, M=Minimize, useADMM=False):
        if useADMM:
            self.__SolveADMM()
            return
        objective = 0
        constraints = []
        ni = TUNGraph.BegNI(self)
        for i in xrange(TUNGraph.GetNodes(self)):
            nid = ni.GetId()
            objective += self.node_objectives[nid]
            constraints += self.node_constraints[nid]
            ni.Next()
        ei = TUNGraph.BegEI(self)
        for i in xrange(TUNGraph.GetEdges(self)):
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            objective += self.edge_objectives[etup]
            constraints += self.edge_constraints[etup]
            ei.Next()
        objective = M(objective)
        problem = Problem(objective, constraints)
        problem.solve()
        self.status = problem.status
        self.value = problem.value

    def __SolveADMM(self):
        self.__SolveADMMDistributed()
        # print 'Solving with ADMM...'
        # # Hash table storing the numpy array values of x, z, and u.
        # admm_node_vals = {}
        # admm_edge_vals = {}
        # (Z_IJ, Z_JI, U_IJ, U_JI) = (0, 1, 2, 3)
        # num_iterations = 50
        # rho = 1.0

        # # Initialize x variable for each node.
        # ni = TUNGraph.BegNI(self)
        # for i in xrange(TUNGraph.GetNodes(self)):
        #     nid = ni.GetId()
        #     varsize = self.node_variables[nid].size
        #     admm_node_vals[nid] = numpy.zeros((varsize[0], varsize[1]))
        #     ni.Next()
        # # Initialize z and u variables for each edge.
        # ei = TUNGraph.BegEI(self)
        # for i in xrange(TUNGraph.GetEdges(self)):
        #     etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
        #     varsize_i = self.node_variables[etup[0]].size
        #     z_ij = numpy.zeros((varsize_i[0], varsize_i[1]))
        #     u_ij = numpy.zeros((varsize_i[0], varsize_i[1]))
        #     varsize_j = self.node_variables[etup[1]].size
        #     z_ji = numpy.zeros((varsize_j[0], varsize_j[1]))
        #     u_ji = numpy.zeros((varsize_j[0], varsize_j[1]))
        #     admm_edge_vals[etup] = [z_ij, z_ji, u_ij, u_ji]
        #     ei.Next()

        # # Run ADMM for a finite number of iterations.
        # # TODO: Stopping conditions.
        # for i in xrange(num_iterations):
        #     # Debugging information prints current iteration #.
        #     print '..%d' % i

        #     # x update: Update x_i with z and u variables constant.
        #     ni = TUNGraph.BegNI(self)
        #     for i in xrange(TUNGraph.GetNodes(self)):
        #         nid = ni.GetId()
        #         var = self.node_variables[nid]
        #         norms = 0
        #         # Sum over all neighbors.
        #         for j in xrange(ni.GetDeg()):
        #             nbrid = ni.GetNbrNId(j)
        #             (zi, ui) = (Z_IJ, U_IJ) if (nid < nbrid) else (Z_JI, U_JI)
        #             edge_vals = admm_edge_vals[self.__GetEdgeTup(nid, nbrid)]
        #             norms += square(norm(var - edge_vals[zi] + edge_vals[ui]))
        #         objective = self.node_objectives[nid] + (rho / 2) * norms
        #         objective = Minimize(objective)
        #         problem = Problem(objective, [])
        #         problem.solve()
        #         admm_node_vals[nid] = var.value
        #         ni.Next()

        #     # z update: Update z_ij and z_ji with x and u variables constant.
        #     ei = TUNGraph.BegEI(self)
        #     for i in xrange(TUNGraph.GetEdges(self)):
        #         etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
        #         edge_vals = admm_edge_vals[etup]
        #         node_val_i = admm_node_vals[etup[0]]
        #         node_val_j = admm_node_vals[etup[1]]
        #         node_var_i = self.node_variables[etup[0]]
        #         node_var_j = self.node_variables[etup[1]]
        #         objective = self.edge_objectives[etup]
        #         o = node_val_i - node_var_i + edge_vals[U_IJ]
        #         objective += (rho / 2) * square(norm(o))
        #         o = node_val_j - node_var_j + edge_vals[U_JI]
        #         objective += (rho / 2) * square(norm(o))
        #         objective = Minimize(objective)
        #         problem = Problem(objective, [])
        #         problem.solve()
        #         # TODO: What if both node variables are not in the edge obj?
        #         edge_vals[Z_IJ] = node_var_i.value
        #         edge_vals[Z_JI] = node_var_j.value
        #         ei.Next()

        #     # u update: Update u with x and z variables constant.
        #     ei = TUNGraph.BegEI(self)
        #     for i in xrange(TUNGraph.GetEdges(self)):
        #         etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
        #         edge_vals = admm_edge_vals[etup]
        #         edge_vals[U_IJ] += admm_node_vals[etup[0]] - edge_vals[Z_IJ]
        #         edge_vals[U_JI] += admm_node_vals[etup[1]] - edge_vals[Z_JI]
        #         ei.Next()

        # self.node_values = admm_node_vals
        # self.status = 'TODO'
        # self.value = 'TODO'

    def __SolveADMMDistributed(self):
        global node_vals, edge_vals, getValue

        print 'Solving with ADMM (DISTRIBUTED)'
        num_processors = 8
        num_iterations = 50
        rho = 1.0

        (X_NID, X_OBJ, X_VAR, X_IND, X_SIZE, X_DEG, X_NEIGHBORS) = range(7)
        node_info = {}
        length = 0
        ni = TUNGraph.BegNI(self)
        for i in xrange(TUNGraph.GetNodes(self)):
            nid = ni.GetId()
            deg = ni.GetDeg()
            obj = self.node_objectives[nid]
            var = self.node_variables[nid]
            varsize = var.size[0]
            neighbors = [ni.GetNbrNId(j) for j in xrange(deg)]
            node_info[nid] = (nid, obj, var, length, varsize, deg, neighbors)
            length += varsize
            ni.Next()
        node_vals = Array('d', [0.0] * length)

        (Z_EID, Z_OBJ, Z_IVAR, Z_ISIZE, Z_XIIND, Z_ZIJIND, Z_UIJIND,\
            Z_JVAR, Z_JSIZE, Z_XJIND, Z_ZJIIND, Z_UJIIND) = range(12)
        edge_list = []
        edge_info = {}
        length = 0
        num_edges = 0
        ei = TUNGraph.BegEI(self)
        for i in xrange(TUNGraph.GetEdges(self)):
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            obj = self.edge_objectives[etup]
            info_i = node_info[etup[0]]
            info_j = node_info[etup[1]]
            ind_zij = length
            length += info_i[X_SIZE]
            ind_uij = length
            length += info_i[X_SIZE]
            ind_zji = length
            length += info_j[X_SIZE]
            ind_uji = length
            length += info_j[X_SIZE]
            tup = (etup, obj,\
                info_i[X_VAR], info_i[X_SIZE], info_i[X_IND], ind_zij, ind_uij,\
                info_j[X_VAR], info_j[X_SIZE], info_j[X_IND], ind_zji, ind_uji)
            edge_list.append(tup)
            num_edges += 1
            edge_info[etup] = tup
            ei.Next()
        edge_vals = Array('d', [0.0] * length)

        node_list = []
        num_nodes = 0
        for nid, info in node_info.iteritems():
            entry = [nid, info[X_OBJ], info[X_VAR], info[X_IND], info[X_SIZE],\
                info[X_DEG]]
            for i in xrange(info[X_DEG]):
                neighborId = info[X_NEIGHBORS][i]
                indices = (Z_ZIJIND, Z_UIJIND) if nid < neighborId else\
                    (Z_ZJIIND, Z_UJIIND)
                einfo = edge_info[self.__GetEdgeTup(nid, neighborId)]
                entry.append(einfo[indices[0]])
                entry.append(einfo[indices[1]])
            node_list.append(entry)
            num_nodes += 1

        pool = Pool(num_processors)
        # Keep track of total time for x-, z-, and u-updates
        totalTimes = [0.0] * 3
        # Keep track of total time for solvers in x- and z-updates
        solverTimes = [0.0] * 2
        # TODO: Stopping conditions.
        for i in xrange(1, num_iterations + 1):
            # Debugging information prints current iteration # and times
            print '..%d' % i

            t0 = time.time()
            time_info = pool.map(ADMM_x, node_list)
            t1 = time.time()
            totalTimes[0] += t1 - t0
            avg_time = sum(time_info) / num_nodes
            solverTimes[0] += avg_time
            print '    x-update: %.5f seconds (Solver Avg: %.5f, Max: %.5f, Min: %.5f' %\
                ((t1 - t0), avg_time, max(time_info), min(time_info))

            t0 = time.time()
            time_info = pool.map(ADMM_z, edge_list)
            t1 = time.time()
            totalTimes[1] += t1 - t0
            avg_time = sum(time_info) / num_edges
            solverTimes[1] += avg_time
            print '    z-update: %.5f seconds (Solver Avg: %.5f, Max: %.5f, Min: %.5f' %\
                ((t1 - t0), avg_time, max(time_info), min(time_info))

            t0 = time.time()
            pool.map(ADMM_u, edge_list)
            t1 = time.time()
            totalTimes[2] += t1 - t0
            print '    u-update: %.5f seconds' % (t1 - t0)
        pool.close()
        pool.join()

        print 'Average x-update: %.5f seconds' % (totalTimes[0] / num_iterations)
        print '     Solver only: %.5f seconds' % (solverTimes[0] / num_iterations)
        print 'Average z-update: %.5f seconds' % (totalTimes[1] / num_iterations)
        print '     Solver only: %.5f seconds' % (solverTimes[1] / num_iterations)
        print 'Average u-update: %.5f seconds' % (totalTimes[2] / num_iterations)

        for entry in node_list:
            nid = entry[X_NID]
            index = entry[X_IND]
            size = entry[X_SIZE]
            self.node_values[nid] = getValue(node_vals, index, size)
        self.status = 'TODO'
        self.value = 'TODO'

    # API to get node variable value after solving with ADMM.
    def GetNodeValue(self, NId):
        self.__VerifyNId(NId)
        return self.node_values[NId] if (NId in self.node_values) else None


    # Helper method to verify existence of an NId.
    def __VerifyNId(self, NId):
        if not TUNGraph.IsNode(self, NId):
            raise Exception('Node %d does not exist.' % NId)

    # Adds a Node to the TUNGraph and stores the corresponding CVX information.
    def AddNode(self, NId, Objective, Variable, Constraints=[]):
        self.node_objectives[NId] = Objective
        self.node_variables[NId] = Variable
        self.node_constraints[NId] = Constraints
        return TUNGraph.AddNode(self, NId)

    def SetNodeObjective(self, NId, Objective):
        self.__VerifyNId(NId)
        self.node_objectives[NId] = Objective

    def GetNodeObjective(self, NId):
        self.__VerifyNId(NId)
        return self.node_objectives[NId]

    def SetNodeVariable(self, NId, Variable):
        self.__VerifyNId(NId)
        self.node_variables[NId] = Variable

    def GetNodeVariable(self, NId):
        self.__VerifyNId(NId)
        return self.node_variables[NId]

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
    def AddEdge(self, SrcNId, DstNId, Objective, Constraints=[]):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.edge_objectives[ETup] = Objective
        self.edge_constraints[ETup] = Constraints
        return TUNGraph.AddEdge(self, SrcNId, DstNId)

    def SetEdgeObjective(self, SrcNId, DstNId, Objective):
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


node_vals = None
edge_vals = None

def getValue(arr, index, size):
    return numpy.array(arr[index:(index + size)])

def writeValue(sharedarr, index, nparr, size):
    if size == 1:
        nparr = [nparr]
    sharedarr[index:(index + size)] = nparr

def ADMM_x(entry):
    # Temporary for now. TODO: Remove.
    (X_NID, X_OBJ, X_VAR, X_IND, X_SIZE, X_DEG, X_NEIGHBORS) = range(7)
    rho = 1.0

    var = entry[X_VAR]
    size = entry[X_SIZE]
    norms = 0
    for i in xrange(entry[X_DEG]):
        z_index = 6 + (2 * i)
        u_index = z_index + 1
        zi = entry[z_index]
        ui = entry[u_index]
        z = getValue(edge_vals, zi, size)
        u = getValue(edge_vals, ui, size)
        norms += square(norm(var - z + u))
    objective = entry[X_OBJ] + (rho / 2) * norms
    objective = Minimize(objective)
    problem = Problem(objective, [])
    t0 = time.time()
    problem.solve()
    t1 = time.time()
    writeValue(node_vals, entry[X_IND], var.value, size)
    return (t1 - t0)

def ADMM_z(entry):
    # Temporary for now. TODO: Remove.
    (Z_EID, Z_OBJ, Z_IVAR, Z_ISIZE, Z_XIIND, Z_ZIJIND, Z_UIJIND,\
        Z_JVAR, Z_JSIZE, Z_XJIND, Z_ZJIIND, Z_UJIIND) = range(12)
    rho = 1.0
    objective = entry[Z_OBJ]

    size_i = entry[Z_ISIZE]
    x_i = getValue(node_vals, entry[Z_XIIND], size_i)
    var_i = entry[Z_IVAR]
    u_ij = getValue(edge_vals, entry[Z_UIJIND], size_i)
    objective += (rho / 2) * square(norm(x_i - var_i + u_ij))

    size_j = entry[Z_JSIZE]
    x_j = getValue(node_vals, entry[Z_XJIND], size_j)
    var_j = entry[Z_JVAR]
    u_ji = getValue(edge_vals, entry[Z_UJIIND], size_j)
    objective += (rho / 2) * square(norm(x_j - var_j + u_ji))

    objective = Minimize(objective)
    problem = Problem(objective, [])
    t0 = time.time()
    problem.solve()
    t1 = time.time()
    writeValue(edge_vals, entry[Z_ZIJIND], var_i.value, size_i)
    writeValue(edge_vals, entry[Z_ZJIIND], var_j.value, size_j)
    return (t1 - t0)

def ADMM_u(entry):
    # Temporary for now. TODO: Remove.
    (Z_EID, Z_OBJ, Z_IVAR, Z_ISIZE, Z_XIIND, Z_ZIJIND, Z_UIJIND,\
        Z_JVAR, Z_JSIZE, Z_XJIND, Z_ZJIIND, Z_UJIIND) = range(12)
    rho = 1.0

    size_i = entry[Z_ISIZE]
    uij = getValue(edge_vals, entry[Z_UIJIND], size_i) +\
          getValue(node_vals, entry[Z_XIIND], size_i) -\
          getValue(edge_vals, entry[Z_ZIJIND], size_i)
    writeValue(edge_vals, entry[Z_UIJIND], uij, size_i)

    size_j = entry[Z_JSIZE]
    uji = getValue(edge_vals, entry[Z_UJIIND], size_j) +\
          getValue(node_vals, entry[Z_XJIND], size_j) -\
          getValue(edge_vals, entry[Z_ZJIIND], size_j)
    writeValue(edge_vals, entry[Z_UJIIND], uji, size_j)
    return entry
