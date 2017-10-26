import numpy as np

from QuadElement import QuadElement
from Material import Material
from Specification import Specification

d = 2 # dimensioality


# specification:
spec = Specification('angledtwoquads')


# sort encastres so that they will be removed from the matrix from right to left
encastres = sorted(spec.encastres, key = lambda enc: enc.point_idx)


n_nodes = spec.nodes.shape[0]
n_elems = spec.elem_nodes.shape[0]




# construct array of elements
elems = []
for i in range(0, n_elems):
    node_indices = np.asarray(spec.elem_nodes)[i]
    node_coords = spec.nodes[np.ix_(node_indices)]
    elems.append(QuadElement(spec.mat, node_coords))


#print(elems[0].getLocalToGlobalCoordsMatrix())
#print("---")

# compute global stiffness matrix
stiffness = np.zeros((n_nodes * d, n_nodes * d))

for i in range(0, n_elems):
    elem = elems[i]
    k = elem.getGlobalStiffnessMatrix()
    #print("K: " + str(k))
    assert(k.shape == (4 * d, 4 * d))
    node_indices = np.asarray(spec.elem_nodes[i])[0]
    coord_indices = np.empty(4 * 2, dtype=int)
    coord_indices[0::2] = node_indices * 2
    coord_indices[1::2] = node_indices * 2 + 1
    stiffness[np.ix_(coord_indices, coord_indices)] += k

'''   
print("---")
print("--- STIFFNESS MATRIX ---")
print(stiffness)
print("---")
print("---")
'''

#Vector as large as the output dispacement vector with values pointing
# to the origin of the displacement value in the output.
#These might differ because applying boundary conditions changes the indexing in the dispacement vector.
out_idx_to_point_coord_idx = np.arange(n_nodes * d)

# apply boundary conditions
load_vector = np.zeros((n_nodes, d))
for load in spec.loads:
    load_vector[load.point_idx,] = [load.x, load.y]
loads = load_vector.reshape(-1, 1)

for encastre in reversed(encastres):
    coord_indices = range(encastre.point_idx * d, encastre.point_idx * d + d)
    stiffness = np.delete(stiffness, coord_indices, 1)
    stiffness = np.delete(stiffness, coord_indices, 0)
    loads = np.delete(loads, coord_indices, 0)
    out_idx_to_point_coord_idx = np.delete(out_idx_to_point_coord_idx, coord_indices, 0)


# SOLVE
displacements = np.linalg.solve(stiffness, loads)

# reinsert encastre points
final_displacements = np.zeros((n_nodes * d, 1))
for i in range(displacements.size):
    final_idx = out_idx_to_point_coord_idx[i]
    final_displacements[final_idx,0] = displacements[i,0]

final_displacements = final_displacements.reshape((-1, d))
    
print(final_displacements)
print("modified: " + str(final_displacements * 500))


'''

t_in = 250# tinner temperature
t_out = 35# air temperature
l1 = .1
l2 = .2
l3 = .15
k1 = .35
k2 = .09
k3 = .9
h = 45


def getLocal(k, l): # get the local stiffness matrix
    mat = np.matrix([[1, -1], [-1, 1]])
    print(mat.dot(-k/l))
    return mat.dot(-k/l)

def combine(existing, to_append): # add two matrices together on the last row and column of the first matrix
    assert(existing.shape[0] == existing.shape[1])
    w = existing.shape[0] + 1 # width and/or height of new matrix
    new_shape = (w, w) # shape of the new matrix
    ret = np.zeros(shape = new_shape)
    ret[0 : w - 1, 0 : w - 1] = existing
    ret[w - 2 : w, w - 2 : w] += to_append
    return ret
    
def applyBC(mat, flux): # convert known t_in into flux vector and drop the row and column from the matrix 
    assert(mat.shape[0] == mat.shape[1])
    w = mat.shape[0] # width and/or height of old matrix
    assert(flux.shape[0] == w)
    # drop first row and column
    ret_mat = mat[1 : w, 1 : w]
    # drop first row of flux vector
    ret_flux = np.zeros((w - 1, 1))
    # convert known t_in into value in flux vector
    # changes 1st row of original from
    # (-k0/l0 k1/l1-k0/l0 ...) (T) = (q1 q2 q3 q4)
    # to (k1/l1-k0/l0 ...) (T) = (q2+k0/l0 q3 q4)
    ret_flux[0 : w - 1] = flux[1 : w]
    ret_flux[0] -= mat[0,1] * t_in
    return (ret_mat, ret_flux)

def applyLoad(mat, flux): # add relation between q3 and t_in
    assert(mat.shape[0] == mat.shape[1])
    w = mat.shape[0] # width and/or height of old matrix
    assert(flux.shape[0] == w)
    ret_mat = mat # copy matrix
    ret_flux = flux # copy load vector
    ret_mat[w-1,w-1] -= h # add influence of T3 on convection
    ret_flux[w - 1] = -h * t_out # add static influence of t_out on convection
    return (ret_mat, ret_flux)

# get the basic matrix and flux vector for the equilibrium equation
m = combine(combine(getLocal(k1, l1), getLocal(k2, l2)), getLocal(k3, l3))
flux = np.zeros((4,1)) # same size as current matrix
# apply boundary conditions
(m, flux) = applyBC(m, flux)
(m, flux) = applyLoad(m, flux)
# solve the system
solution = np.linalg.solve(m, flux)
print(m)
print(flux)
print(solution)

# abaqus output:
#    250. 
# 227.222
# 50.0589
# 36.7716


'''