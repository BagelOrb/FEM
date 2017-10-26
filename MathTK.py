import numpy as np
import math # sqrt
import functools # reduce

class MathTK:
    def __init__(self):
        pass
        
    @staticmethod
    def doGaussLegendreIntegration(f, degrees):
        (sampling_points, weights) = np.polynomial.legendre.leggauss(degrees)
        sum = 0
        for i in range(sampling_points.size):
            sum += weights[i] * f(sampling_points[i])
        return sum
    
    @staticmethod
    def do2DimensionalGaussLegendreIntegration(f, degrees):
        (sampling_points, weights) = np.polynomial.legendre.leggauss(degrees)
        sum = 0
        for x in range(sampling_points.size):
            for y in range(sampling_points.size):
                sum += weights[x] * weights[y] * f(sampling_points[x], sampling_points[y])
        return sum
    
    # convert matrix of coords to vector of lengths
    @staticmethod
    def computeMagnitudes(final_displacements):
        n_nodes = final_displacements.shape[1]
        magnitude = np.zeros((n_nodes, 1))
        for i in range(n_nodes):
            displacement_coords = np.asarray(final_displacements)[i]
            magnitude[i] = math.sqrt(functools.reduce(lambda sum, coord: sum + coord * coord, displacement_coords, 0))
        return magnitude