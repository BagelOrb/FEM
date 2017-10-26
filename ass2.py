import numpy as np

from QuadElement import QuadElement
from Material import Material


mat = Material(youngs_modulus = 70000, poisson_ratio = 0.3)

n_dims = 2;

nodes = np.matrix([ [10, 10],
                    [11, 10],
                    [11, 11],
                    [10, 11],
                    [12, 10],
                    [12, 11]
                    ]) # TODO

n_nodes = nodes.shape[0]

elem_nodes = np.matrix([[0,1,2,3]
                        #, [1,4,5,2]
                        ])

n_elems = elem_nodes.shape[0]

# construct array of elements
elems = []
for i in range(0, n_elems):
    node_indices = np.asarray(elem_nodes)[i]
    node_coords = nodes[np.ix_(node_indices)]
    elems.append(QuadElement(mat, node_coords))

print("=====================")

print(elems[0].getLocalToGlobalCoordsMatrix())
print("---")


stiffness = np.zeros((n_nodes * n_dims, n_nodes * n_dims))
#stiffness = np.arange(n_nodes * n_dims * n_nodes * n_dims).reshape(n_nodes * n_dims, n_nodes * n_dims)

for i in range(0, n_elems):
    elem = elems[i]
    k = elem.getGlobalStiffnessMatrix()
    assert(k.shape == (4 * n_dims, 4 * n_dims))
    node_indices = np.asarray(elem_nodes[i])[0]
    coord_indices = np.empty(4 * 2, dtype=int)
    coord_indices[0::2] = node_indices * 2
    coord_indices[1::2] = node_indices * 2 + 1
    stiffness[np.ix_(coord_indices, coord_indices)] += k
    
print("---")
print(elems[0].getGlobalStiffnessMatrix())
print("---")
print(stiffness)


print("---")
print("---")
print("---")

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