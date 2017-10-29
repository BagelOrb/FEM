import numpy as np
import collections

#define boundary condition types
Pinned = collections.namedtuple('Pinned', 'point_idx')
Symmetry = collections.namedtuple('Symmetry', 'point_idx dir')
Load = collections.namedtuple('Load', 'point_idx x y')

integration_steps = 3

### Specification ###

f = 1000
thickness = 2

E = 70000 # youngs_modulus
mu = 0.3 # poissons ratio
D = np.matrix([ [1 , mu, 0           ], # strain_to_stress matrix
                [mu, 1 , 0           ],
                [0 , 0 , (1 - mu) / 2]]) * (E / (1 - mu * mu))

nodes = np.matrix([ [    -50.,         340.],
                    [    -50.,         440.],
                    [   -200.,         440.],
                    [   -200.,         340.],
                    [    -50.,     -109.177],
                    [ 60.9004,     -127.299],
                    [    250.,         440.],
                    [    -50.,        -360.],
                    [-18.2857,     -364.857],
                    [   -200.,        -360.],
                    [   -200.,        -460.],
                    [    -50.,        -460.],
                    [ 6.76268,     -110.907],
                    [ 97.2326,     -18.3021],
                    [ 140.052,      110.156],
                    [ 190.519,      261.556],
                    [ 89.6865,         440.],
                    [    -50.,      145.268],
                    [    -50., -324.038E-03],
                    [    -50.,     -214.606],
                    [    -50.,       -297.8],
                    [-34.1429,     -362.429],
                    [ 3.91181,     -298.265],
                    [ 30.0715,     -219.786],
                    [ -106.46,        -360.],
                    [-154.523,        -360.],
                    [   -200.,     -410.015],
                    [-159.052,        -460.],
                    [-120.031,        -460.],
                    [-86.1814,        -460.],
                    [-62.7363,        -460.],
                    [ -35.443,     -416.329],
                    [ 25.0065,      2.14856],
                    [ 46.0295,      138.124],
                    [ 69.0587,       294.92],
                    [-8.85942,     -209.312],
                    [-22.2726,     -291.951],
                    [-52.0233,     -412.347],
                    [-75.7638,     -410.629],
                    [-114.841,     -410.169],
                    [-157.104,     -410.046]
                    ])

elem_nodes = np.matrix([    [1 ,2 ,3 ,4 ], # 0
                            [5 ,13,33,19], # 1
                            [6 ,13,33,14], # 2
                            [18,19,33,34], # 3
                            [14,15,34,33], # 4
                            [1 ,18,34,35], # 5
                            [15,16,35,34], # 6
                            [1 ,2 ,17,35], # 7
                            [7 ,16,35,17], # 8
                            [6 ,13,36,24], # 9
                            [5 ,13,36,20], # 10
                            [23,24,36,37], # 11
                            [20,21,37,36], # 12
                            [9 ,22,37,23], # 13
                            [8 ,21,37,22], # 14
                            [9 ,22,38,32], # 15
                            [8 ,22,38,39], # 16
                            [8 ,25,40,39], # 17
                            [25,26,41,40], # 18
                            [10,26,41,27], # 19
                            [12,31,38,32], # 20
                            [30,31,38,39], # 21
                            [29,30,39,40], # 22
                            [28,29,40,41], # 23
                            [11,27,41,28]  # 24
                            ]) - 1 # abaqus indexing starts at 1 because they are not programmers

loads = [Load(point_idx = 10, x=0, y=-f)]

boundary_conditions = [   Pinned(point_idx = 1)
                        , Pinned(point_idx = 2)
                        , Pinned(point_idx = 6)
                        , Pinned(point_idx = 16)
                        , Symmetry(point_idx = 3, dir = 0)
                        , Symmetry(point_idx = 9, dir = 0)
                        , Symmetry(point_idx = 10, dir = 0)
                        , Symmetry(point_idx = 26, dir = 0)]

n_nodes = nodes.shape[0]
n_elems = elem_nodes.shape[0]

dim_count = 2 # dimensions
d = 2 # dimensions
n = 4 # nodes in a quad

### main function for calculating local stiffness matrices ###
def toBeIntegrated(global_coords, r, s):
    shape_function_derivatives_matrix = np.matrix([
        [-(1-s),0,1-s,0,(1+s),0,-(1+s),0],
        [-(1-r),0,-(1+r),0,(1+r),0,1-r,0],
        [0,-(1-s),0,1-s,0,(1+s),0,-(1+s)],
        [0,-(1-r),0,-(1+r),0,(1+r),0,1-r]]) * .25
    dxy_drs = shape_function_derivatives_matrix.dot(global_coords)
    det_j = np.linalg.det(dxy_drs.reshape(2,2).transpose())
    strain_disp_derivatives = np.matrix([   [ dxy_drs[3,0],-dxy_drs[2,0],      0      ,            0], # ( du/dr )
                                            [      0      ,      0      ,-dxy_drs[1,0], dxy_drs[0,0]], # ( du/ds )
                                            [-dxy_drs[1,0], dxy_drs[0,0], dxy_drs[3,0],-dxy_drs[2,0]]  # ( dv/ds )
                                            ])  * (1 / det_j)                                          # ( dv/dr )
    B = strain_disp_derivatives.dot(shape_function_derivatives_matrix)
    return B.transpose().dot(D).dot(B) * (det_j * thickness)

### calculate Global Stiffness Matrix ###
stiffness = np.zeros((n_nodes * d, n_nodes * d))
for i in range(0, n_elems):
    node_indices = np.asarray(elem_nodes)[i]
    node_coords = nodes[np.ix_(node_indices)]
    global_coords = node_coords.reshape(-1,1)

    k = np.zeros((4 * d, 4 * d))
    (sampling_points, weights) = np.polynomial.legendre.leggauss(integration_steps)
    for x in range(sampling_points.size):
        for y in range(sampling_points.size):
            k += weights[x] * weights[y] * toBeIntegrated(global_coords, sampling_points[x], sampling_points[y])
    coord_indices = np.empty(4 * 2, dtype=int)
    coord_indices[0::2] = node_indices * 2
    coord_indices[1::2] = node_indices * 2 + 1
    stiffness[np.ix_(coord_indices, coord_indices)] += k

### APPLY Boundary Conditions ###

#Vector as large as the output dispacement vector with values pointing
# to the origin of the displacement value in the output.
#These might differ because applying boundary conditions changes the indexing in the dispacement vector.
out_idx_to_point_coord_idx = np.arange(n_nodes * d)

load_vector = np.zeros((n_nodes, d))
for load in loads:
    load_vector[load.point_idx,] = [load.x, load.y]
loads = load_vector.reshape(-1, 1)


# sort boundary_conditions so that they will be removed from the matrix from right to left
boundary_conditions = sorted(boundary_conditions, key = lambda enc: enc.point_idx)
for boundary_condition in reversed(boundary_conditions):
    if isinstance(boundary_condition, Pinned):
        coord_indices = range(boundary_condition.point_idx * d, boundary_condition.point_idx * d + d)
    elif isinstance(boundary_condition, Symmetry):
        coord_indices = boundary_condition.point_idx * d + boundary_condition.dir
    stiffness = np.delete(stiffness, coord_indices, 1) # delete column
    stiffness = np.delete(stiffness, coord_indices, 0) # delete row
    loads = np.delete(loads, coord_indices, 0)
    out_idx_to_point_coord_idx = np.delete(out_idx_to_point_coord_idx, coord_indices, 0)

### SOLVE ###
displacements = np.linalg.solve(stiffness, loads)

# reinsert boundary_condition points
final_displacements = np.zeros((n_nodes * d, 1))
for i in range(displacements.size):
    final_idx = out_idx_to_point_coord_idx[i]
    final_displacements[final_idx,0] = displacements[i,0]

final_displacements = final_displacements.reshape((-1, d))
print(final_displacements)
