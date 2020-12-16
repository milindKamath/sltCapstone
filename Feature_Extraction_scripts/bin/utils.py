import os, glob, sys, re
import yaml
import pandas as pd
import numpy as np

import torch
from torchvision import transforms
from PIL import Image

import pdb

############################################################################
# alexnet model
def model_alexnet( use_cuda ):
    # get alexnet model
    model = torch.hub.load( 'pytorch/vision:v0.6.0', 'alexnet', pretrained=True )
    classifier = torch.nn.Sequential( *list( model.classifier.children() )[:-1] )
    model.classifier = classifier
    model.eval()
    if use_cuda:
        model.cuda()
    return model

############################################################################
# resnet50 model
def model_resnet50( use_cuda ):
    # get alexnet model
    model = torch.hub.load( 'pytorch/vision:v0.6.0', 'resnet50', pretrained=True )
    classifier = torch.nn.Sequential( *list( model.children() )[:-1] )
    model = classifier
    model.eval()
    if use_cuda:
        model.cuda()
    return model

############################################################################
# extract features
def extract_features( paths, model, batch_size, use_cuda ):
    all_features = np.array([])
    preprocess = transforms.Compose([ transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize( mean=[ 0.485,0.456,0.406 ],
                                                             std=[0.229, 0.224, 0.225]
                                                           )
                                    ])
    for i in range( 0, len( paths ), batch_size ):
        temp = paths[ i:i+batch_size ]
        imgs = torch.Tensor()
        for filepath in temp:
            # read image
            img = Image.open(filepath)
            # preprocess image
            input_tensor = preprocess( img )
            # get right dims
            input_tensor = input_tensor.unsqueeze(0)
            imgs = input_tensor if imgs.ndim == 1 else torch.cat( (imgs, input_tensor) )

        if use_cuda:
            imgs = imgs.cuda()
        with torch.no_grad():
            # extract features
            features = model( imgs )
            features = torch.squeeze( features )
        features = features.cpu().numpy()
        all_features = features if all_features.size==0 else np.vstack( (all_features, features) )
    return all_features

############################################################################
# load config file
def load_config( path ):
    if not os.path.exists( path ):
        print( 'No config file present.' )
        sys.exit(1)
    with open( path, 'r' ) as f:
        conf = yaml.safe_load( f )
    return conf

############################################################################
# get all image paths
def get_img_paths( df, vid_col, path_col, gloss_col=None, ext='jpg', sort_type='default' ):
    ignore = ['hand', 'body']
    img_paths = {}
    for index, row in df.iterrows():
        video_name = row[ vid_col ]
        path = row[ path_col ]
        # for glosses
        frames = None
        if gloss_col:
            frames = row[ gloss_col ]
            frames = None if isinstance( frames, float ) else frames.split(',')
        
        ########################################################################
        # get all paths
        all = []
        filepaths = glob.glob( f'{path}/*{ext}' )
        for filepath in filepaths:
            name, _ = os.path.splitext( filepath.split('/')[-1] )
            if [ ele for ele in ignore if re.search( ele, name ) ]:
                continue
            # all files
            all.append( filepath )

        if len(all) == 0:
            print( 'Could not find all img files for {video_name}...')
            continue
        else:
            res = all

        ########################################################################
        gloss = []
        if frames:
            for fr in frames:
                filepath = os.path.join( path, f'{fr}.{ext}' )
                if filepath in filepaths:
                    gloss.append( filepath )
            if len(gloss) > 0:
                res = gloss

        ########################################################################
        # sort all and gloss
        if sort_type == 'default':
            res.sort()
        else:
            res = sorted( res, key=lambda x: int( os.path.splitext( x.split('/')[-1] )[0] ) )
        img_paths[ video_name ] = res

    return img_paths

############################################################################
# check if output already exists
def check_output_exists( out_dir, img_paths ):
    res = {}
    for vid, paths in img_paths.items():
        out_path = f'{out_dir}/{vid}.npy'
        if not os.path.exists( out_path ):
            res[vid] = paths
    return res