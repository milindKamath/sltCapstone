import argparse, os, sys, re
import pandas as pd
import glob

from utils import load_config

############################################################################
# add path to the video
def get_vid_paths( x, data, additional_files, additional_folder ):
    if x in additional_files:
        vid_path = os.path.join( additional_folder, x )
    else:
        vid_path = os.path.join( data, x )
    return vid_path

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
    # get all video paths and add the col
    data = conf['data']
    additional_files = conf['additional_files']
    additional_folder = conf['additional_folder']
    vid_col = conf['vid_col']
    path_col = conf['path_col']
    df[path_col] = df[vid_col].apply( lambda x: get_vid_paths( x, data, additional_files, additional_folder ) )

    ############################################################################
    # check if images exists for the video otherwise drop it 
    ignore = ['hand', 'body']
    indexes = []
    for index, row in df.iterrows():
        vid = row[vid_col]
        path = row[path_col]
        temp = None
        for filepath in glob.glob( f'{path}/*.jpg' ):
            filename = filepath.split('/')[-1]
            name, _ = os.path.splitext( filename )
            if [ ele for ele in ignore if re.search( ele, name ) ]:
                continue
            temp = filepath
            break
        if temp == None:
            indexes.append( index )

    if len(indexes) > 0:
        df = df.drop( indexes ).reset_index( drop=True )
    print( f'Total number of videos that have images: {df.shape[0]}' )

    ############################################################################
    # if number of samples then generate csv for those samples only
    samples = conf.get( 'samples', None )
    if samples:
        df = df.sample( n=int( samples ), replace=False, random_state=0 ).reset_index( drop=True )
    ############################################################################
    # save the csv in the output path specified
    out = conf['out']
    df.to_csv( out, index=False )
    print( f'Total number of videos that are saved in the csv: {df.shape[0]}' )
    
if __name__ == "__main__":
    main()