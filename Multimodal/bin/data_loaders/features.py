import os
import pandas as pd
import json
import numpy as np
import torch

import pickle
import gzip 

import pdb

############################################################################
# parent class
class Feature:
    def __init__( self, set, dir, name, add_value=0.0 ):
        self.name = name
        self.set = set
        self.dir = dir
        self.add_value = add_value
            
    def load_feature( self, name ):
        filepath = f'{self.dir}/{name}.npy'
        fts = np.load( filepath ).astype( np.float32 )
        fts = torch.tensor( fts ) + self.add_value
        if self.name == 'optical_flow':
            fts[0] = fts[1]
        
        self.dims = fts.shape[-1]
        return fts

    def __len__( self ):
        return self.dims

############################################################################
# Kmeans features
class Kmeans( Feature ):
    def __init__(self, set, dir, vid_col='video_name', ft_col='cluster_ids', name='kmeans', sl='gsl' ):
        self.set = set
        self.dir = dir
        output = f'{dir}/{set}'
        if not os.path.exists( output ):
            os.mkdir( output )
            csv = f'{dir}/kmeans_{self.set}.csv'
            vid_col = vid_col
            ft_col = ft_col
            df = pd.read_csv( csv )
            for index, row in df.iterrows():
                vid_name = row[vid_col]
                fts = row[ft_col]
                fts = json.loads( fts ) 
                fts = np.array( fts, dtype=np.float32 )
                fts = np.expand_dims( fts, axis=1 )
                out_path = f'{output}/{vid_name}.npy'
                np.save( out_path, fts )
        self.dir = f'{dir}/{self.set}'
        super().__init__( self.set, self.dir, name, 1.0 )

############################################################################
# Joeynmt features
class Joeynmt( Feature ):
    def __init__( self, set, dir, name='joeynmt', sl='gsl' ):
        if set == 'val':
            self.set = 'dev'
        else:
            self.set = set
        self.dir = dir
        output = f'{self.dir}/{self.set}'
        if not os.path.exists( output ):
            pickle_file = f'{self.dir}/phoenix14t.pami0.{self.set}'
            with gzip.open( pickle_file, 'rb' ) as f:
                obj = pickle.load(f)
            os.mkdir( output )
            for d in obj:
                vid_name = d['name'].split('/')[-1]
                sign = d['sign']
                sign = sign.numpy()
                out_path = f'{output}/{vid_name}.npy'
                np.save( out_path, sign )

        self.dir = f'{self.dir}/{self.set}'
        super().__init__( self.set, self.dir, name, 1e-8 )


############################################################################
# Resnet50 features
class Resnet50( Feature ):
    def __init__( self, set, dir, name='resnet50', sl='gsl' ):
        if sl == 'asl':
            self.set = set
            self.dir = dir
        else:
            if set == 'val':
                self.set = 'dev'
            else:
                self.set = set
            self.dir = f'{dir}/{self.set}'

        super().__init__( self.set, self.dir, name, 1e-8 )
    

############################################################################
# Alexnet features
class Alexnet( Feature ):
    def __init__( self, set, dir, name='alexnet', sl='gsl' ):
        if sl == 'asl':
            self.set = set
            self.dir = dir
        else:
            if set == 'val':
                self.set = 'dev'
            else:
                self.set = set
            self.dir = f'{dir}/{self.set}'
        super().__init__( self.set, self.dir, name, 1e-8 )

############################################################################
# Openpose features
class Openpose( Feature ):
    def __init__( self, set, dir, name='openpose', sl='gsl' ):
        if sl == 'asl':
            self.set = set
            self.dir = dir
        else:
            if set == 'val':
                self.set = 'dev'
            else:
                self.set = set
            self.dir = f'{dir}/{self.set}'
        super().__init__( self.set, self.dir, name, 1e-8 )

############################################################################
# Optical Flow
class Optical_Flow( Feature ):
    def __init__( self, set, dir, name='optical_flow', sl='gsl' ):
        if sl == 'asl':
            self.set = set
            self.dir = dir
        else:
            if set == 'val':
                self.set = 'dev'
            else:
                self.set = set
            self.dir = f'{dir}/{self.set}'

        super().__init__( self.set, self.dir, name, 1e-8 )

sets = {
    'kmeans': Kmeans,
    'joeynmt': Joeynmt,
    'openpose': Openpose,
    'resnet50': Resnet50,
    'alexnet': Alexnet,
    'optical_flow': Optical_Flow
}