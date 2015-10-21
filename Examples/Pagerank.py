from snapvx import *
import numpy as np

# Build a random graph
np.random.seed(1)
num_nodes = 10
num_edges = 25
snapGraph =GenRndGnm(PUNGraph, num_nodes, num_edges)
gvx=TGraphVX(snapGraph)

# Define edge weights
c = dict()
d_list =[0] * num_nodes
for ei in gvx.Edges():
    (src_id, dst_id) = (ei.GetSrcNId(), ei.GetDstNId())   
    weight = np.random.uniform(0.5,1)
    c[(src_id, dst_id)] = weight
    (d_list[src_id],d_list[dst_id]) = (d_list[src_id] + weight, d_list[dst_id] + weight)

# Define parameters to solve pagerank problem
beta=Parameter(sign="positive",value = 0.85) # transition probability
alpha = 1/beta.value -1
v=Parameter(num_nodes,sign="positive") # Random restart distribution
val=np.random.uniform(size=num_nodes) # Use np.ones if all equally likely
v.value = val/sum(val)


#------------ Solve Using SnapVX (approach based on: http://jmlr.org/proceedings/papers/v32/gleich14.pdf) --------------
# Set node objectives
for ni in gvx.Nodes():
    n_id=ni.GetId()
    x = Variable(1,name='x')
    d=d_list[n_id]
    gvx.SetNodeObjective(n_id,alpha*v.value[n_id]*square(1-x) + alpha*(d-v.value[n_id])*square(x))

# For each edge in the original graph, define objective function
for ei in gvx.Edges():    
    (src_id, dst_id) = (ei.GetSrcNId(), ei.GetDstNId())     
    (src_vars, dst_vars) = (gvx.GetNodeVariables(src_id), gvx.GetNodeVariables(dst_id))
    edge_obj=  c[(src_id, dst_id)]*square(src_vars['x']-dst_vars['x'])
    gvx.SetEdgeObjective(src_id,dst_id,edge_obj)

gvx.Solve(UseADMM=False) # If UseADMM=True, set EpsAbs and EpsRel to 0.0002 for better convergence

# Print the renormalized solution
print '\nSolution via ||B(v)x||_{C(s),2} (using SnapVX)'
for ni in gvx.Nodes():
    n_id=ni.GetId()
    if n_id< num_nodes:
        print 'Node %d: %f' %(n_id, d_list[n_id]*gvx.GetNodeValue(n_id,'x'))


#--------------------- Compare to Normal Pagerank -------------------------------
A=np.zeros([num_nodes,num_nodes])
# Generate original Adjacency matrix
for ei in gvx.Edges():
    (src_id, dst_id) = (ei.GetSrcNId(), ei.GetDstNId())
    if (src_id < num_nodes) & (dst_id < num_nodes): # For edge not linking to s or t node
        (A[src_id, dst_id], A[dst_id, src_id]) = (c[(src_id, dst_id)], c[(src_id, dst_id)])

B=(1+alpha)*np.diag(d_list)-A
z= np.linalg.solve(B, alpha*v.value)
# Print the renormalized solution
print '\nSolution via (I-AD^{-1})x=beta*v'
for n_id in range(num_nodes):
    print 'Node %d: %f' %(n_id,d_list[n_id]*z[n_id])

