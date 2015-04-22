## snapvx

from snap import *
from cvxpy import *
from multiprocessing import *
import numpy
import time


def LoadEdgeList(filename):
    gvx = TGraphVX()
    nids = set()
    infile = open(filename, 'r')
    for line in infile:
        if line.startswith('#'): continue
        [src, dst] = line.split()
        if int(src) not in nids:
            gvx.AddNode(int(src))
        if int(dst) not in nids:
            gvx.AddNode(int(dst))
        gvx.AddEdge(int(src), int(dst))
    return gvx


class TGraphVX(TUNGraph):

    __default_objective = Minimize(0)
    __default_constraints = []

    # node_objectives  = {int NId : CVXPY Expression}
    # node_constraints = {int NId : [CVXPY Constraint]}
    # edge_objectives  = {(int NId1, int NId2) : CVXPY Expression}
    # edge_constraints = {(int NId1, int NId2) : [CVXPY Constraint]}
    #
    # ADMM-Specific Structures
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
        self.node_objectives = {}
        self.node_variables = {}
        self.node_constraints = {}
        self.edge_objectives = {}
        self.edge_constraints = {}
        self.node_values = {}

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


    # Iterates through all nodes and edges. Currently adds objectives together.
    # Option of specifying Maximize() or the default Minimize().
    # Option to use serial version or distributed ADMM.
    # Graph status and value properties will be set in serial version.
    # Individual variable values can always be retrieved using GetNodeValue().
    def Solve(self, M=Minimize, useADMM=True, rho=1.0):
        if useADMM:
            self.__SolveADMM(rho)
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
        # Insert into hash to match ADMMDistributed() output
        ni = TUNGraph.BegNI(self)
        for i in xrange(TUNGraph.GetNodes(self)):
            nid = ni.GetId()
            variables = self.node_variables[nid]
            value = numpy.array([])
            for (varID, varName, var, offset) in variables:
                val = var.value
                if var.size[0] == 1:
                    val = numpy.array([val])
                value = numpy.concatenate((value, val))
            self.node_values[nid] = value
            ni.Next()

    def __SolveADMM(self, rho_param):
        global node_vals, edge_vals, rho, getValue

        num_processors = 8
        num_iterations = 50
        rho = rho_param
        print 'Solving with distributed ADMM (%d processors)' % num_processors

        # Node ID, CVXPY Objective, CVXPY Variables, CVXPY Constraints,
        #   Starting index into node_vals, Length of all variables, Node degree
        (X_NID, X_OBJ, X_VARS, X_CON, X_IND, X_LEN, X_DEG, X_NEIGHBORS) = range(8)
        node_info = {}
        length = 0
        ni = TUNGraph.BegNI(self)
        for i in xrange(TUNGraph.GetNodes(self)):
            nid = ni.GetId()
            deg = ni.GetDeg()
            obj = self.node_objectives[nid]
            variables = self.node_variables[nid]
            con = self.node_constraints[nid]
            neighbors = [ni.GetNbrNId(j) for j in xrange(deg)]
            for neighborId in neighbors:
                etup = self.__GetEdgeTup(nid, neighborId)
                econ = self.edge_constraints[etup]
                con += econ
            l = 0
            for (varID, varName, var, offset) in variables:
                l += var.size[0]
            node_info[nid] = (nid, obj, variables, con, length, l, deg, neighbors)
            length += l
            ni.Next()
        node_vals = Array('d', [0.0] * length)

        (Z_EID, Z_OBJ, Z_CON, Z_IVARS, Z_ILEN, Z_XIIND, Z_ZIJIND, Z_UIJIND,\
            Z_JVARS, Z_JLEN, Z_XJIND, Z_ZJIIND, Z_UJIIND) = range(13)
        edge_list = []
        edge_info = {}
        length = 0
        num_edges = 0
        ei = TUNGraph.BegEI(self)
        for i in xrange(TUNGraph.GetEdges(self)):
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            obj = self.edge_objectives[etup]
            con = self.edge_constraints[etup]
            con += self.node_constraints[etup[0]] + self.node_constraints[etup[1]]
            info_i = node_info[etup[0]]
            info_j = node_info[etup[1]]
            ind_zij = length
            length += info_i[X_LEN]
            ind_uij = length
            length += info_i[X_LEN]
            ind_zji = length
            length += info_j[X_LEN]
            ind_uji = length
            length += info_j[X_LEN]
            tup = (etup, obj, con,\
                info_i[X_VARS], info_i[X_LEN], info_i[X_IND], ind_zij, ind_uij,\
                info_j[X_VARS], info_j[X_LEN], info_j[X_IND], ind_zji, ind_uji)
            edge_list.append(tup)
            num_edges += 1
            edge_info[etup] = tup
            ei.Next()
        edge_vals = Array('d', [0.0] * length)

        node_list = []
        num_nodes = 0
        for nid, info in node_info.iteritems():
            entry = [nid, info[X_OBJ], info[X_VARS], info[X_CON], info[X_IND], info[X_LEN],\
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
            # Debugging information prints current iteration #
            print '..%d' % i

            t0 = time.time()
            time_info = pool.map(ADMM_x, node_list)
            t1 = time.time()
            totalTimes[0] += t1 - t0
            avg_time = sum(time_info) / num_nodes
            solverTimes[0] += avg_time
            # print '    x-update: %.5f seconds (Solver Avg: %.5f, Max: %.5f, Min: %.5f' %\
            #     ((t1 - t0), avg_time, max(time_info), min(time_info))

            t0 = time.time()
            time_info = pool.map(ADMM_z, edge_list)
            t1 = time.time()
            totalTimes[1] += t1 - t0
            avg_time = sum(time_info) / num_edges
            solverTimes[1] += avg_time
            # print '    z-update: %.5f seconds (Solver Avg: %.5f, Max: %.5f, Min: %.5f' %\
            #     ((t1 - t0), avg_time, max(time_info), min(time_info))

            t0 = time.time()
            pool.map(ADMM_u, edge_list)
            t1 = time.time()
            totalTimes[2] += t1 - t0
            # print '    u-update: %.5f seconds' % (t1 - t0)
        pool.close()
        pool.join()

        # print 'Average x-update: %.5f seconds' % (totalTimes[0] / num_iterations)
        # print '     Solver only: %.5f seconds' % (solverTimes[0] / num_iterations)
        # print 'Average z-update: %.5f seconds' % (totalTimes[1] / num_iterations)
        # print '     Solver only: %.5f seconds' % (solverTimes[1] / num_iterations)
        # print 'Average u-update: %.5f seconds' % (totalTimes[2] / num_iterations)

        for entry in node_list:
            nid = entry[X_NID]
            index = entry[X_IND]
            size = entry[X_LEN]
            self.node_values[nid] = getValue(node_vals, index, size)
        self.status = 'TODO'
        self.value = 'TODO'

    # API to get node variable value after solving with ADMM.
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
        if filename == None:
            ni = TUNGraph.BegNI(self)
            for i in xrange(TUNGraph.GetNodes(self)):
                nid = ni.GetId()
                print 'Node %d:' % nid
                for (varID, varName, var, offset) in self.node_variables[nid]:
                    print ' ', varName, numpy.transpose(self.GetNodeValue(nid, varName))
                ni.Next()
        else:
            outfile = open(filename, 'w+')
            ni = TUNGraph.BegNI(self)
            for i in xrange(TUNGraph.GetNodes(self)):
                nid = ni.GetId()
                s = 'Node %d:\n' % nid
                outfile.write(s)
                for (varID, varName, var, offset) in self.node_variables[nid]:
                    s = '  %s %s\n' % (varName, str(numpy.transpose(self.GetNodeValue(nid, varName))))
                    outfile.write(s)
                ni.Next()


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
    def AddNode(self, NId, Objective=__default_objective, Constraints=__default_constraints):
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
    def AddEdge(self, SrcNId, DstNId, Objective=__default_objective, Constraints=__default_constraints):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
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


    # Get dictionary of all variables corresponding to a node.
    # { string name : CVXPY Variable }
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
    # node ID for each row. The index of this column (0-indexed) is idColumn.
    def AddNodeObjectives(self, filename, obj_func, nodeIDs=None, idColumn=0):
        infile = open(filename, 'r')
        if nodeIDs == None:
            for line in infile:
                if line.startswith('#'): continue
                data = [x.strip() for x in line.split(',')]
                ret = obj_func(data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetNodeObjective(int(data[idColumn]), ret[0])
                    self.SetNodeConstraints(int(data[idColumn]), ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetNodeObjective(int(data[idColumn]), ret)
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
    # obj_func is a function which accepts two arguments, a dictionary of
    #     variables for the source and destination nodes
    #     { string varName : CVXPY Variable }
    # obj_func should return a tuple of (objective, constraints), although
    #     it will assume a singleton object will be an objective
    def AddEdgeObjectives(self, obj_func):
        ei = TUNGraph.BegEI(self)
        for i in xrange(TUNGraph.GetEdges(self)):
            src_id = ei.GetSrcNId()
            src_vars = self.GetNodeVariables(src_id)
            dst_id = ei.GetDstNId()
            dst_vars = self.GetNodeVariables(dst_id)
            ret = obj_func(src_vars, dst_vars)
            if type(ret) is tuple:
                # Tuple = assume we have (objective, constraints)
                self.SetEdgeObjective(src_id, dst_id, ret[0])
                self.SetEdgeConstraints(src_id, dst_id, ret[1])
            else:
                # Singleton object = assume it is the objective
                self.SetEdgeObjective(src_id, dst_id, ret)
            ei.Next()


node_vals = None
edge_vals = None
rho = 1.0

def getValue(arr, index, size):
    return numpy.array(arr[index:(index + size)])

def writeValue(sharedarr, index, nparr, size):
    if size == 1:
        nparr = [nparr]
    sharedarr[index:(index + size)] = nparr

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

def ADMM_x(entry):
    global rho
    # Temporary for now. TODO: Remove.
    (X_NID, X_OBJ, X_VARS, X_CON, X_IND, X_LEN, X_DEG, X_NEIGHBORS) = range(8)

    variables = entry[X_VARS]
    norms = 0
    for i in xrange(entry[X_DEG]):
        z_index = X_NEIGHBORS + (2 * i)
        u_index = z_index + 1
        zi = entry[z_index]
        ui = entry[u_index]
        # Add norm for each variable corresponding to the node
        for (varID, varName, var, offset) in variables:
            z = getValue(edge_vals, zi + offset, var.size[0])
            u = getValue(edge_vals, ui + offset, var.size[0])
            norms += square(norm(var - z + u))

    objective = entry[X_OBJ] + (rho / 2) * norms
    objective = Minimize(objective)
    constraints = entry[X_CON]
    problem = Problem(objective, constraints)
    t0 = time.time()
    problem.solve()
    t1 = time.time()

    writeObjective(node_vals, entry[X_IND], objective, variables)
    return (t1 - t0)

def ADMM_z(entry):
    global rho
    # Temporary for now. TODO: Remove.
    (Z_EID, Z_OBJ, Z_CON, Z_IVARS, Z_ILEN, Z_XIIND, Z_ZIJIND, Z_UIJIND,\
        Z_JVARS, Z_JLEN, Z_XJIND, Z_ZJIIND, Z_UJIIND) = range(13)
    objective = entry[Z_OBJ]
    constraints = entry[Z_CON]
    norms = 0

    variables_i = entry[Z_IVARS]
    for (varID, varName, var, offset) in variables_i:
        x_i = getValue(node_vals, entry[Z_XIIND] + offset, var.size[0])
        u_ij = getValue(edge_vals, entry[Z_UIJIND] + offset, var.size[0])
        norms += square(norm(x_i - var + u_ij))

    variables_j = entry[Z_JVARS]
    for (varID, varName, var, offset) in variables_j:
        x_j = getValue(node_vals, entry[Z_XJIND] + offset, var.size[0])
        u_ji = getValue(edge_vals, entry[Z_UJIIND] + offset, var.size[0])
        norms += square(norm(x_j - var + u_ji))

    objective = Minimize(objective + (rho / 2) * norms)
    problem = Problem(objective, constraints)
    t0 = time.time()
    problem.solve()
    t1 = time.time()

    writeObjective(edge_vals, entry[Z_ZIJIND], objective, variables_i)
    writeObjective(edge_vals, entry[Z_ZJIIND], objective, variables_j)
    return (t1 - t0)

def ADMM_u(entry):
    global rho
    # Temporary for now. TODO: Remove.
    (Z_EID, Z_OBJ, Z_CON, Z_IVARS, Z_ILEN, Z_XIIND, Z_ZIJIND, Z_UIJIND,\
        Z_JVARS, Z_JLEN, Z_XJIND, Z_ZJIIND, Z_UJIIND) = range(13)

    size_i = entry[Z_ILEN]
    uij = getValue(edge_vals, entry[Z_UIJIND], size_i) +\
          getValue(node_vals, entry[Z_XIIND], size_i) -\
          getValue(edge_vals, entry[Z_ZIJIND], size_i)
    writeValue(edge_vals, entry[Z_UIJIND], uij, size_i)

    size_j = entry[Z_JLEN]
    uji = getValue(edge_vals, entry[Z_UJIIND], size_j) +\
          getValue(node_vals, entry[Z_XJIND], size_j) -\
          getValue(edge_vals, entry[Z_ZJIIND], size_j)
    writeValue(edge_vals, entry[Z_UJIIND], uji, size_j)
    return entry
