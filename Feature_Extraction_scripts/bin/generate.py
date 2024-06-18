import os, sys, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
import concurrent.futures

from utils import load_config, get_img_paths, check_output_exists
from utils import model_alexnet, model_resnet50
from utils import extract_features

import pdb

models = {
    'alexnet': model_alexnet,
    'resnet50': model_resnet50
}

def main():
    ############################################################################
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( 'config', help='configuration file' )
    args = parser.parse_args()

    ############################################################################
    # load config
    conf = load_config( args.config )

    ############################################################################
    # read csv
    csv = conf['csv']
    df = pd.read_csv( csv )
    print( f'Total number of videos in csv: {df.shape[0]}' )

    ############################################################################
    # get image paths in dict
    vid_col = conf['vid_col']
    path_col = conf['path_col']
    gloss_col = conf.get( 'gloss_col', None )
    ext = conf.get( 'ext', 'jpg' )
    sort_type = conf.get( 'sort_type', 'default' )
    img_paths = get_img_paths( df, vid_col, path_col, gloss_col=gloss_col, ext=ext, sort_type=sort_type )
    print( f'Total number of videos found from csv: {len(img_paths)}' )

    ############################################################################
    # check with output if output already exists
    out_dir = conf['out_dir']
    new_img_paths = check_output_exists( out_dir, img_paths )
    print( f'Total number of videos already processed: {len(img_paths)-len(new_img_paths)}' )
    print( f'Total number of videos to process: {len(new_img_paths.keys())}' )
    ############################################################################
    # get feature
    use_cuda = conf['use_cuda']
    feature = conf['feature']
    

    ############################################################################
    # generate features for each video and store them
    batch_size = conf['batch_size']

    if feature == 'optical_flow':
        func = models.get( 'resnet50' )
        model = func( use_cuda )
        ############################################################################
        # import optical flow functions [ requires cv2 ]
        from optical_flow import extract_optical_flow, get_pairs, add_duplicates

        ############################################################################
        optical_flow_out = conf['optical_flow_out']
        if not os.path.exists( optical_flow_out ):
            os.mkdir( optical_flow_out )
        for key, values in tqdm( new_img_paths.items(), total=len(new_img_paths), desc='Extracting ->' ):
            of_out_dir = os.path.join( optical_flow_out, key )
            if not os.path.exists( of_out_dir ):
                os.mkdir( of_out_dir )

            ############################################################################
            # get optical flow features
            params = ( key, of_out_dir )
            fn = partial( extract_optical_flow, params )
            pairs = get_pairs( values )
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list( executor.map( fn, pairs )  )

            results = add_duplicates( values, results )
            ############################################################################
            # extract resnet features and save them
            features = extract_features( results, model, batch_size, use_cuda )
            save_features_path = os.path.join( out_dir, f'{key}.npy' )
            np.save( save_features_path, features )
    else:
        func = models.get( feature )
        model = func( use_cuda )
        for vid, paths in tqdm( new_img_paths.items(), desc='Extracting ->', total=len(new_img_paths) ):
            features = extract_features( paths, model, batch_size, use_cuda )
            out_path = f'{out_dir}/{vid}.npy'
            np.save( out_path, features )

    print( f'Shape of the extracted feature: {features.shape[-1]}' )
    print( 'Features for all videos extracted' )

    
if __name__ == "__main__":
    main()