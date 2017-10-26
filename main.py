import numpy as np

from Specification import Specification
from MathTK import MathTK
from FEM import FEM 

#main:
fem = FEM(Specification('angledonequad'))
displacements = fem.solve()
print("displacements: " + str(displacements))
magnitudes = MathTK.computeMagnitudes(displacements)
print("magnitudes: " + str(magnitudes))
