import numpy as np
from MathTK import MathTK

class QuadElement:
    dim_count = 2
    elem_count = 4
    integration_steps = 3 # number of boxes of discretization in the approximation of an integral when using Gauss-Legendre Integration
    def __init__(self, material, global_coords):
        d = self.dim_count
        n = self.elem_count
        self.material = material
        self.global_coords = global_coords.reshape(-1,1)
            
    def getGlobalStiffnessMatrix(self):
        result = MathTK.do2DimensionalGaussLegendreIntegration(self.toBeIntegrated, self.integration_steps)
        return result

    
    def toBeIntegrated(self, r, s):
        B = self.getB(r, s)
        D = self.material.D
        det = self.det_J
        t = self.material.thickness
        return B.transpose().dot(D).dot(B) * (det * t)
    
    def getB(self, r, s):
        shape_function_derivatives = self.getShapeFunctionDerivatives(r, s)
        M = self.getM(shape_function_derivatives)
        jacobian = self.getJacobian(M)
        #print("shape_function_derivatives:" + str(shape_function_derivatives))
        #print("coords: " +str(self.global_coords))
        #print("M:" + str(M))
        #print("j: " + str(jacobian))
        A = self.getA(jacobian)
        return A.dot(M)
    
    # the jacobian for the strain
    #
    # argument jacobian:
    # [ dx/dr  dy/dr ]
    # [ dx/ds  dy/ds ]
    def getA(self, jacobian):
        d = QuadElement.dim_count
        n = QuadElement.elem_count
        j = jacobian
        assert(j.shape == (d, d))
        #print("J: " + str(j))
        #print("det_J: " +str(det_J))
        return np.matrix([  [ j[1,1], -j[0,1],       0,       0], # ( du/dr )
                            [      0,       0, -j[1,0],  j[0,0]], # ( du/ds )
                            [-j[1,0],  j[0,0],  j[1,1], -j[0,1]]  # ( dv/ds )
                            ])  * (1 / self.det_J)               # ( dv/dr )

    def getJacobian(self, M):
        d = QuadElement.dim_count
        derivatives = M.dot(self.global_coords)
        # dx/dr
        # dx/ds
        # dy/dr
        # dy/ds
        ret = derivatives.reshape(d, d).transpose()
        self.det_J = np.linalg.det(ret)
        return ret
    
    @staticmethod
    def getM(shape_function_derivatives):
        d = QuadElement.dim_count
        n = QuadElement.elem_count
        shape_function_derivatives
        ret = np.zeros((d * d, n * d))
        for xydim in range(d):
            for rsdim in range(d):
                ret[xydim * d + rsdim, xydim::d] = shape_function_derivatives[rsdim]
                #              v v v v v v v v v v v v v
                #
                # ( du/dr )   [ dN1dr   0   dN2dr   0   ] ( u1_x )
                # ( du/ds ) = [ dN1ds   0   dN2ds   0   ] ( u1_y )
                # ( dv/ds )   [   0   dN1dr   0   dN2dr ] ( u2_x )
                # ( dv/dr )   [   0   dN1ds   0   dN2ds ] ( u2_y )
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
