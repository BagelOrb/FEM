import numpy as np
import math # sqrt
import functools # reduce

from QuadElement import QuadElement
from Material import Material
from Specification import Specification

d = 2 # dimensioality
class FEM:

    def __init__(self, spec):
        self.spec = spec
        self.n_nodes = self.spec.nodes.shape[0]
        self.n_elems = self.spec.elem_nodes.shape[0]
        
        self.constructElems()
        self.computeGlobalStifnessMatrix()
        self.applyLoadsAndBoundaryConditions()


    # construct array of elements
    def constructElems(self):
        self.elems = []
        for i in range(0, self.n_elems):
            node_indices = np.asarray(self.spec.elem_nodes)[i]
            node_coords = self.spec.nodes[np.ix_(node_indices)]
            self.elems.append(QuadElement(self.spec.mat, node_coords))

    # verify element shape functions
    def verifyElems(self):
        for elem in self.elems:
            print(str(elem.getXY(-1,-1)[0,0])+", "+str(elem.getXY(-1,-1)[1,0]) +" == "+ str(elem.global_coords[0,0]) +", "+ str(elem.global_coords[1,0]))
            print(str(elem.getXY( 1,-1)[0,0])+", "+str(elem.getXY( 1,-1)[1,0]) +" == "+ str(elem.global_coords[2,0]) +", "+ str(elem.global_coords[3,0]))
            print(str(elem.getXY( 1, 1)[0,0])+", "+str(elem.getXY( 1, 1)[1,0]) +" == "+ str(elem.global_coords[4,0]) +", "+ str(elem.global_coords[5,0]))
            print(str(elem.getXY(-1, 1)[0,0])+", "+str(elem.getXY(-1, 1)[1,0]) +" == "+ str(elem.global_coords[6,0]) +", "+ str(elem.global_coords[7,0]))

    
    # compute global stiffness matrix
    def computeGlobalStifnessMatrix(self):
        self.stiffness = np.zeros((self.n_nodes * d, self.n_nodes * d))

        for i in range(0, self.n_elems):
            elem = self.elems[i]
            k = elem.getGlobalStiffnessMatrix()
            #print("K: " + str(k))
            assert(k.shape == (4 * d, 4 * d))
            node_indices = np.asarray(self.spec.elem_nodes[i])[0]
            coord_indices = np.empty(4 * 2, dtype=int)
            coord_indices[0::2] = node_indices * 2
            coord_indices[1::2] = node_indices * 2 + 1
            self.stiffness[np.ix_(coord_indices, coord_indices)] += k

    def applyLoadsAndBoundaryConditions(self):
        #Vector as large as the output dispacement vector with values pointing
        # to the origin of the displacement value in the output.
        #These might differ because applying boundary conditions changes the indexing in the dispacement vector.
        self.out_idx_to_point_coord_idx = np.arange(self.n_nodes * d)

        # apply boundary conditions
        load_vector = np.zeros((self.n_nodes, d))
        for load in self.spec.loads:
            load_vector[load.point_idx,] = [load.x, load.y]
        self.loads = load_vector.reshape(-1, 1)


        # sort encastres so that they will be removed from the matrix from right to left
        encastres = sorted(self.spec.encastres, key = lambda enc: enc.point_idx)
        for encastre in reversed(encastres):
            coord_indices = range(encastre.point_idx * d, encastre.point_idx * d + d)
            self.stiffness = np.delete(self.stiffness, coord_indices, 1) # delete column
            self.stiffness = np.delete(self.stiffness, coord_indices, 0) # delete row
            self.loads = np.delete(self.loads, coord_indices, 0)
            self.out_idx_to_point_coord_idx = np.delete(self.out_idx_to_point_coord_idx, coord_indices, 0)


    def solve(self):
        displacements = np.linalg.solve(self.stiffness, self.loads)

        # reinsert encastre points
        final_displacements = np.zeros((self.n_nodes * d, 1))
        for i in range(displacements.size):
            final_idx = self.out_idx_to_point_coord_idx[i]
            final_displacements[final_idx,0] = displacements[i,0]

        final_displacements = final_displacements.reshape((-1, d))
        return final_displacements
