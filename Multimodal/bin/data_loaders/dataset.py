import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

import pandas as pd
from pattern.en import lexeme
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from data_loaders.features import sets
from data_loaders.features import Kmeans, Joeynmt, Openpose, Resnet50, Alexnet

import pdb

############################################################################
# Multifeature data class
class Multifeature_dataset( Dataset ):
    def __init__( self, csv, vid_col, caption_col, vocab_obj, f1_obj, f2_obj=None, f3_obj=None, sep=',' ):
        self.csv = csv
        self.vocab_obj = vocab_obj
        self.f1_obj = f1_obj
        self.f2_obj = f2_obj
        self.f3_obj = f3_obj
        self.vid_col = vid_col
        self.caption_col = caption_col

        # dataframe 
        self.df = pd.read_csv( self.csv, sep=sep )

    def __len__(self):
        return self.df.shape[0]

    def get_dim( self ):
        vid_name = self.df[ self.vid_col ].iloc[0]
        _ = self.f1_obj.load_feature( vid_name )
        if self.f2_obj is not None and self.f3_obj is not None:
            _ = self.f2_obj.load_feature( vid_name )
            _ = self.f3_obj.load_feature( vid_name )
            return len( self.f1_obj ), len( self.f2_obj ), len( self.f3_obj )
        else:
            return len(self.f1_obj)
        

    def __getitem__(self, index):
        vid_name = self.df[ self.vid_col ].iloc[ index ]
        # obtain features
        f1_feature = self.f1_obj.load_feature( vid_name )
        if self.f2_obj is not None and self.f2_obj is not None:
            f2_feature = self.f2_obj.load_feature( vid_name )
            f3_feature = self.f3_obj.load_feature( vid_name )
            X = ( f1_feature, f2_feature, f3_feature )
        else:
            X = ( f1_feature )
        # obtain label
        caption = self.df[ self.caption_col ].iloc[ index ]
        caption, ids = self.vocab_obj.get_caption_ids( caption )
        y = torch.tensor( ids )
        sgn_length = f1_feature.shape[0]
        return X, y, caption, sgn_length, vid_name

############################################################################
# collate func for padding and masking
def my_collate_fn( data ):
    X, y, caption, sgn_lengths, vid_name = zip(*data)
    f1, f2, f3 = None, None, None
    try:
        f1, f2, f3 = zip(*X)
    except:
        f1 = X
    f1 = pad_sequence( f1, batch_first=True, padding_value=0 )
    if  f2 and f3:
        f2 = pad_sequence( f2, batch_first=True, padding_value=0 )
        f3 = pad_sequence( f3, batch_first=True, padding_value=0 )
        X = ( f1, f2, f3 )
    else:
        X = ( f1 )

    # get src mask for each features
    # f1_mask = ( f1 != torch.ones( f1.shape[-1] ) )[..., 0].unsqueeze(1)   # (b, 1, seq)
    f1_mask = ( f1 != torch.zeros( f1.shape[-1] ) )[..., 0].unsqueeze(1)

    # pad captions
    y = pad_sequence( y, batch_first=True, padding_value=1 )

    # txt_input
    txt_input = y[:, :-1] 
    # get trg mask
    txt_mask = ( txt_input != 1 ).unsqueeze(1)

    # txt
    txt = y[:, 1:]

    y = ( txt_input, txt, caption )
    masks = ( f1_mask, txt_mask )

    sgn_lengths = torch.tensor( sgn_lengths, dtype=torch.float32 )

    return X, y, masks, sgn_lengths, vid_name


############################################################################
# data loader
def data_loader( set, data, model, vocab_obj ):
    # features
    features = data['features']
    sl = data['sl']
    f1_obj, f2_obj, f3_obj = None, None, None
    f1_obj = sets.get( features['f1']['name'] )( set, features['f1']['path'], sl=sl )
    if 'f2' in features.keys() and 'f3' in features.keys():
        f2_obj = sets.get( features['f2']['name'] )( set, features['f2']['path'], sl=sl )
        f3_obj = sets.get( features['f3']['name'] )( set, features['f3']['path'], sl=sl )

    # sep for reading csv
    if sl == 'gsl':
        sep = '|'
    else:
        sep = ','

    # other params
    csv = data[ set ]
    vid_col = data['vid_col']
    caption_col = data['caption_col']
    batch_size = model['batch_size']
    shuffle = model['shuffle']
    dataset = Multifeature_dataset( csv, vid_col, caption_col, vocab_obj, f1_obj, f2_obj, f3_obj, sep=sep )
    size = len( dataset )
    dims = dataset.get_dim()
    loader = DataLoader( dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate_fn )
    return loader, size, dims