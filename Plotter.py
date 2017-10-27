import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt

from FEM import FEM
from QuadElement import QuadElement

class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
    
    def run(self):
        self.ax.set_title('Displacements')

        self.ax.autoscale_view()
        
    @staticmethod
    def plot(spec, displacements):
        plotter = Plotter()
        
        max_displacement = np.max(displacements)
        bb = np.max(spec.nodes) - np.min(spec.nodes)
        scaling_factor = abs(bb / max_displacement / 100)
        print("scaling_factor: " + str(scaling_factor))
        displaced_nodes = spec.nodes + displacements * scaling_factor
        
        plotter.plotQuads(spec.nodes, spec.elem_nodes, 'green')
        plotter.plotQuads(displaced_nodes, spec.elem_nodes, 'red')
        
        plotter.run()


        plt.show()
        
    def plotQuads(self, nodes, elem_nodes, color, output_elem_idx = False, output_node_idx=False):
        n_elems = elem_nodes.shape[0]
        vertices = np.zeros((n_elems * 5, 2))
        codes = []
        for i in range(n_elems):
            node_indices = np.asarray(elem_nodes)[i]
            node_coords = nodes[np.ix_(node_indices)]
            #print("coords: \n" +str(node_coords))
            if output_elem_idx:
                self.ax.text(np.average(node_coords[0:4,0]), np.average(node_coords[0:4,1]), str(i), horizontalalignment='center', verticalalignment='center',)
            vertices[i * 5 : i * 5 + 4, 0 : 2] = np.asarray(node_coords)
            vertices[i * 5 + 4, 0 : 2] = np.asarray(node_coords[0])
            codes += [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]

        if output_node_idx:
            for i in range(nodes.shape[0]):
                self.ax.text(nodes[i,0], nodes[i,1], str(i), horizontalalignment='center', verticalalignment='center',)
            
        path = Path(vertices, codes)

        pathpatch = PathPatch(path, facecolor='None', edgecolor=color)
        self.ax.add_patch(pathpatch)
        self.ax.dataLim.update_from_data_xy(vertices)
