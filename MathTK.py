import numpy as np

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