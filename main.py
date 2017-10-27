import numpy as np

from Specification import Specification
from MathTK import MathTK
from FEM import FEM 
from Plotter import Plotter

#main:
def do(name):
    print("\n\n---- " + name + " ----\n")
    spec = Specification(name)
    fem = FEM(spec)
    displacements = fem.solve()
    print("displacements: \n" + str(displacements))
    magnitudes = MathTK.computeMagnitudes(displacements)
    #print("magnitudes: \n" + str(magnitudes))
    #Plotter.plotSpec(spec)
    Plotter.plot(spec, displacements)

#do('angledonequad')
#do('onequad')

#do('angledtwoquads')
#do('twoquads')

#do('quadrow')
do('ass2_try2_simple')


