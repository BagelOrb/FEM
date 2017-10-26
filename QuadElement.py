import numpy as np
from MathTK import MathTK

class QuadElement:
    dim_count = 2
    elem_count = 4
    def __init__(self, material, global_coords):
        d = self.dim_count
        n = self.elem_count
        self.material = material
        self.global_coords = global_coords.reshape(-1,1)
        self.T_nodal = np.matrix([[1,2],[3,4]]) # TODO: make depend on coords in the quad!
        self.K = np.identity(d*n) # local stiffness matrix
        print("init")
        self.getJacobian(1,1)
        print("--\\init")
    
    # gives matrix to convert local x1,y1,x2,y2,x3,y3,x4,y4 into global
    def getLocalToGlobalCoordsMatrix(self):
        d = self.dim_count
        n = self.elem_count
        T_elem = np.identity(n * d);
        for i in range(0, n):
            T_elem[i * d : i * d + d, i * d : i * d + d] = self.T_nodal
        return T_elem
    
    def getGlobalStiffnessMatrix(self):
        T_elem = self.getLocalToGlobalCoordsMatrix()
        return T_elem.transpose().dot(self.K).dot(T_elem)
    
    
    def getB(self, r, s):
        jacobian = self.getJacobian(r, s)
        A = self.getA(jacobian)
        shape_function_derivatives = self.getShapeFunctionDerivatives(r, s)
        M = self.getM(shape_function_derivatives)
        return A.dot(M)
    
    # the jacobian for the strain
    #
    # argument jacobian:
    # [ dx/dr  dy/dr ]
    # [ dx/ds  dy/ds ]
    @staticmethod
    def getA(jacobian):
        d = QuadElement.dim_count
        n = QuadElement.elem_count
        j = jacobian
        assert(j.shape == (d, d))
        det = np.linalg.det(j)
        print("J: " + str(j))
        print("det: " +str(det))
        return np.matrix([[j[1,1], -j[0,1], 0, 0],
                            [0, 0, -j[1,0], j[0,0]],
                            [-j[1,0], j[0,0], j[1,1], j[0,1]]
                            ]) * (1 / det)
    
    # not 100% sure
    def getJacobian(self, r, s):
        d = QuadElement.dim_count
        shape_function_derivatives = QuadElement.getShapeFunctionDerivatives(r, s)
        M = QuadElement.getM(shape_function_derivatives)
        derivatives = M.dot(self.global_coords)
        ret = derivatives.reshape(d, d)
        print("coords: " +str(self.global_coords))
        print("shape_function_derivatives:" + str(shape_function_derivatives))
        print("M:" + str(M))
        print("j: " + str(ret))
        return ret
    
    @staticmethod
    def getM(shape_function_derivatives):
        d = QuadElement.dim_count
        n = QuadElement.elem_count
        (dNdr, dNds) = shape_function_derivatives
        ret = np.zeros((d * d, n * d))
        for xydim in range(d):
            ret[xydim * d + 0, xydim::d] = dNdr
            ret[xydim * d + 1, xydim::d] = dNds
        return ret
    
    @staticmethod
    def getShapeFunctionDerivatives(r, s):
        dNdr = []
        dNdr.append(-.25 * (1 - s))
        dNdr.append(.25 * (1 - s))
        dNdr.append(.25 * (1 + s))
        dNdr.append(-.25 * (1 + s))
        dNds = []
        dNds.append(-.25 * (1 - r))
        dNds.append(-.25 * (1 + r))
        dNds.append(.25 * (1 + r))
        dNds.append(.25 * (1 - r))
        return (dNdr, dNds)
        
    
    def getXY(self, r, s):
        d = self.dim_count
        n = self.elem_count
        shape_functions = self.getShapeFunctions(r, s)
        shape_matrix = np.zeros((d, n * d))
        for dim in range(d):
            shape_matrix[dim, dim::d] = np.matrix(shape_functions)
        return shape_matrix.dot(self.global_coords)
        

    @staticmethod
    def getShapeFunctions(r, s):
        ret = []
        ret.append( .25 * (1 - r) * (1 - s))
        ret.append( .25 * (1 + r) * (1 - s))
        ret.append( .25 * (1 + r) * (1 + s))
        ret.append( .25 * (1 - r) * (1 + s))
        return ret
        
    @staticmethod
    def getIsoparametricShapeFunctionMatrix(shape_functions):
        d = self.dim_count
        n = self.elem_count
        ret = np.zeros(d, d * n)
        for x in range(n):
            for y in range(d):
                ret[d, n * d + d] = shape_functions[n]
        return ret