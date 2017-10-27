import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt

from FEM import FEM
from QuadElement import QuadElement
import Specification


class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
    
    def run(self):

        self.ax.autoscale_view()
        
    @staticmethod
    def plot(spec, displacements):
        plotter = Plotter()
        plotter.ax.set_title('Displacements')
        
        max_displacement = np.max(displacements)
        bb = np.max(spec.nodes) - np.min(spec.nodes)
        scaling_factor = abs(bb / max_displacement / 100)
        print("scaling_factor: " + str(scaling_factor))
        displaced_nodes = spec.nodes + displacements * scaling_factor
        
        plotter.plotQuads(spec.nodes, spec.elem_nodes, 'green')
        plotter.plotQuads(displaced_nodes, spec.elem_nodes, 'red')
        
        plotter.run()


        plt.show()
    
    @staticmethod
    def plotSpec(spec):
        plotter = Plotter()
        plotter.ax.set_title('Mesh')
        
        plotter.plotQuads(spec.nodes, spec.elem_nodes, 'green', output_elem_idx = True, output_node_idx=True)
        plotter.plotBCs(spec)
        
        
        plotter.run()


        plt.show()
        
    def plotBCs(self, spec):
        d = 2
        n_elems = spec.nodes.shape[0]
        DOFs = np.array([True] * (n_elems * 2))

        n_DOFs = n_elems * 2
        
        
        bb = np.max(spec.nodes) - np.min(spec.nodes)
        vec_length = bb / 50

        # sort boundary_conditions so that they will be removed from the matrix from right to left
        boundary_conditions = spec.boundary_conditions
        for boundary_condition in reversed(boundary_conditions):
            if isinstance(boundary_condition, Specification.Pinned):
                coord_indices = range(boundary_condition.point_idx * d, boundary_condition.point_idx * d + d)
                n_DOFs -= d
            elif isinstance(boundary_condition, Specification.Symmetry):
                coord_indices = boundary_condition.point_idx * d + boundary_condition.dir
                n_DOFs -= 1
            else:
                raise ValueError("Unknown boundary condition type.")
            
            DOFs[coord_indices] = False
        
        vertices = np.zeros((n_DOFs * 2, 2))
        codes = []
        dof_idx = 0
        for node_idx in range(n_elems):
            if (DOFs[node_idx * 2]):
                codes += [Path.MOVETO] + [Path.LINETO]
                coord = np.asarray(spec.nodes)[node_idx]
                vertices[dof_idx, 0 : 2] = coord
                vertices[dof_idx + 1, 0 : 2] = coord + [vec_length, 0]
                dof_idx += 2
            if (DOFs[node_idx * 2 + 1]):
                codes += [Path.MOVETO] + [Path.LINETO]
                coord = np.asarray(spec.nodes)[node_idx]
                vertices[dof_idx, 0 : 2] = coord
                vertices[dof_idx + 1, 0 : 2] = coord + [0, vec_length]
                dof_idx += 2
                
        path = Path(vertices, codes)

        pathpatch = PathPatch(path, facecolor='None', edgecolor='red', lw=2)
        self.ax.add_patch(pathpatch)


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
