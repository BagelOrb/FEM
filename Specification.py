import numpy as np
import collections

from QuadElement import QuadElement
from Material import Material


#define Encastre type
Encastre = collections.namedtuple('Encastre', 'point_idx')
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
            
            self.encastres = [Encastre(point_idx = 3), Encastre(point_idx = 0)]
            
        elif preset == 'angledtwoquads':
            self.nodes = np.matrix([[0, 0],
                                    [85, 52],
                                    [32, 136],
                                    [-52, 84],
                                    [170, 105],
                                    [117, 189]
                                    ])
            
            self.elem_nodes = np.matrix([[0,1,2,3]
                                        , [1,4,5,2]
                                        ])
            
            self.loads = [Load(point_idx = 5, x=f*.85, y=f*.52)]
            
            self.encastres = [Encastre(point_idx = 3), Encastre(point_idx = 0)]
            
        elif preset == 'onequad':
            self.nodes = np.matrix([  [000, 000]
                                    , [100, 000]
                                    , [100, 100]
                                    , [000, 100]
                                    ])
            
            self.elem_nodes = np.matrix([[0,1,2,3]
                                        ])
            self.loads = [Load(point_idx = 2, x=f, y=0)]
            
            self.encastres = [Encastre(point_idx = 3), Encastre(point_idx = 0)]
            
        elif preset == 'angledonequad':
            self.nodes = np.matrix([[0, 0],
                                    [85, 52],
                                    [32, 136],
                                    [-52, 84],
                                    ])
            
            self.elem_nodes = np.matrix([[0,1,2,3]
                                        ])
            self.loads = [Load(point_idx = 2, x=f*.85, y=f*.52)]
            
            self.encastres = [Encastre(point_idx = 3), Encastre(point_idx = 0)]
            
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
            
            self.elem_nodes = np.matrix([ [1,2,17,0]
                                        , [2,3,16,17]
                                        , [3,4,15,16]
                                        , [4,5,14,15]
                                        , [5,6,13,14]
                                        , [6,7,12,13]
                                        , [7,8,11,12]
                                        , [8,9,10,11]
                                        ])
            
            self.loads = [Load(point_idx = 9, x=f, y=f)]
            
            self.encastres = [Encastre(point_idx = 0), Encastre(point_idx = 1)]
            
        else:
            raise ValueError("Unknown Specification preset name")