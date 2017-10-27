import numpy as np

from Specification import Specification
from MathTK import MathTK
from FEM import FEM 

#main:
def do(name):
    print("\n\n---- " + name + " ----\n")
    fem = FEM(Specification(name))
    displacements = fem.solve()
    print("displacements: \n" + str(displacements))
    magnitudes = MathTK.computeMagnitudes(displacements)
    print("magnitudes: \n" + str(magnitudes))

#do('angledonequad')
#do('onequad')

do('ass2_try2_simple')

