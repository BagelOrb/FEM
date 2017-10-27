import numpy as np
import collections

from QuadElement import QuadElement
from Material import Material


#define Pinned type
Pinned = collections.namedtuple('Pinned', 'point_idx')
Symmetry = collections.namedtuple('Symmetry', 'point_idx dir')
Load = collections.namedtuple('Load', 'point_idx x y')

class Specification:
    def __init__(self, preset):
        self.mat = Material(youngs_modulus = 70000, poisson_ratio = 0.3, thickness = 2)
            
        f = 1000
        
        if preset == 'twoquads':
            self.nodes = np.matrix([[000, 000],
                                    [100, 000],
                                    [100, 100],
                                    [000, 100],
                                    [200, 000],
                                    [200, 100]
                                    ])
            
            self.elem_nodes = np.matrix([[0,1,2,3]
                                        , [1,4,5,2]
                                        ])
            
            self.loads = [Load(point_idx = 5, x=f, y=0)]
            
            self.boundary_conditions = [Pinned(point_idx = 3), Pinned(point_idx = 0)]
            
        elif preset == 'angledtwoquads':
            self.nodes = np.matrix([[0, 0],
                                    [84.811, 52.989],
                                    [32.042, 137.437],
                                    [-52.763, 84],
                                    [169.260, 105.758],
                                    [116.491, 190.207]
                                    ])
            
            self.elem_nodes = np.matrix([[0,1,2,3]
                                        , [1,4,5,2]
                                        ])
            
            self.loads = [Load(point_idx = 5, x=f*.84811, y=f*.52989)]
            
            self.boundary_conditions = [Pinned(point_idx = 3), Pinned(point_idx = 0)]
            
        elif preset == 'onequad':
            self.nodes = np.matrix([  [000, 000]
                                    , [100, 000]
                                    , [100, 100]
                                    , [000, 100]
                                    ])
            
            self.elem_nodes = np.matrix([[0,1,2,3]
                                        ])
            self.loads = [Load(point_idx = 2, x=f, y=0)]
            
            self.boundary_conditions = [Pinned(point_idx = 3), Pinned(point_idx = 0)]
            
        elif preset == 'angledonequad':
            self.nodes = np.matrix([[0, 0],
                                    [85, 52],
                                    [32, 136],
                                    [-52, 84],
                                    ])
            
            self.elem_nodes = np.matrix([[0,1,2,3]
                                        ])
            self.loads = [Load(point_idx = 2, x=f*.85, y=f*.52)]
            
            self.boundary_conditions = [Pinned(point_idx = 3), Pinned(point_idx = 0)]
            
        elif preset == 'quadrow':
            self.nodes = np.matrix([  
                                      [000, 100] # 0
                                    , [000, 000] # 1
                                    , [100, 000] # 2
                                    , [200, 000] # 3
                                    , [300, 000] # 4
                                    , [400, 000] # 5
                                    , [500, 000] # 6
                                    , [600, 000] # 7
                                    , [700, 000] # 8
                                    , [800, 000] # 9
                                    , [800, 100] # 10
                                    , [700, 100] # 11
                                    , [600, 100] # 12
                                    , [500, 100] # 13
                                    , [400, 100] # 14
                                    , [300, 100] # 15
                                    , [200, 100] # 16
                                    , [100, 100] # 17
                                    ])
            
            self.elem_nodes = np.matrix([ [1,2,17,0]  # 1
                                        , [2,3,16,17] # 2
                                        , [3,4,15,16] # 3
                                        , [4,5,14,15] # 4
                                        , [5,6,13,14] # 5
                                        , [6,7,12,13] # 6
                                        , [7,8,11,12] # 7
                                        , [8,9,10,11] # 8
                                        ])
            
            self.loads = [Load(point_idx = 9, x=f, y=f)]
            
            self.boundary_conditions = [Pinned(point_idx = 0), Pinned(point_idx = 1)]
            
        elif preset == 'ass2_try2_simple':
            self.nodes = np.matrix([[    -50.,         340.],
                                    [    -50.,         440.],
                                    [   -200.,         440.],
                                    [   -200.,         340.],
                                    [    -50.,     -109.177],
                                    [ 60.9004,     -127.299],
                                    [    250.,         440.],
                                    [    -50.,        -360.],
                                    [-18.2857,     -364.857],
                                    [   -200.,        -360.],
                                    [   -200.,        -460.],
                                    [    -50.,        -460.],
                                    [ 6.76268,     -110.907],
                                    [ 97.2326,     -18.3021],
                                    [ 140.052,      110.156],
                                    [ 190.519,      261.556],
                                    [ 89.6865,         440.],
                                    [    -50.,      145.268],
                                    [    -50., -324.038E-03],
                                    [    -50.,     -214.606],
                                    [    -50.,       -297.8],
                                    [-34.1429,     -362.429],
                                    [ 3.91181,     -298.265],
                                    [ 30.0715,     -219.786],
                                    [ -106.46,        -360.],
                                    [-154.523,        -360.],
                                    [   -200.,     -410.015],
                                    [-159.052,        -460.],
                                    [-120.031,        -460.],
                                    [-86.1814,        -460.],
                                    [-62.7363,        -460.],
                                    [ -35.443,     -416.329],
                                    [ 25.0065,      2.14856],
                                    [ 46.0295,      138.124],
                                    [ 69.0587,       294.92],
                                    [-8.85942,     -209.312],
                                    [-22.2726,     -291.951],
                                    [-52.0233,     -412.347],
                                    [-75.7638,     -410.629],
                                    [-114.841,     -410.169],
                                    [-157.104,     -410.046]
                                    ])
            
            self.elem_nodes = np.matrix([   [1 ,2 ,3 ,4 ], # 0
                                            [5 ,13,33,19], # 1
                                            [6 ,13,33,14], # 2
                                            [18,19,33,34], # 3
                                            [14,15,34,33], # 4
                                            [1 ,18,34,35], # 5
                                            [15,16,35,34], # 6
                                            [1 ,2 ,17,35], # 7
                                            [7 ,16,35,17], # 8
                                            [6 ,13,36,24], # 9
                                            [5 ,13,36,20], # 10
                                            [23,24,36,37], # 11
                                            [20,21,37,36], # 12
                                            [9 ,22,37,23], # 13
                                            [8 ,21,37,22], # 14
                                            [9 ,22,38,32], # 15
                                            [8 ,22,38,39], # 16
                                            [8 ,25,40,39], # 17
                                            [25,26,41,40], # 18
                                            [10,26,41,27], # 19
                                            [12,31,38,32], # 20
                                            [30,31,38,39], # 21
                                            [29,30,39,40], # 22
                                            [28,29,40,41], # 23
                                            [11,27,41,28]  # 24
                                        ]) - 1 # abaqus indexing starts at 1 because they are not programmers
            
            self.loads = [Load(point_idx = 10, x=0, y=-f)]
            
            self.boundary_conditions = [  Pinned(point_idx = 1)
                                        , Pinned(point_idx = 2)
                                        , Pinned(point_idx = 6)
                                        , Pinned(point_idx = 16)
                                        , Symmetry(point_idx = 3, dir = 0)
                                        , Symmetry(point_idx = 9, dir = 0)
                                        , Symmetry(point_idx = 10, dir = 0)
                                        , Symmetry(point_idx = 26, dir = 0)
                                        ]

        
        
        else:
            raise ValueError("Unknown Specification preset name")
