from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory

import os, sys, argparse
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from nltk.tokenize import word_tokenize

sys.path.append( './bin' )
from src.utils import load_config
from src.metrics import bleu

import pdb

app = Flask( __name__ )

def score( row, vid_col, ft_test ):
    row = row.copy()
    vid = row[vid_col]
    ft = os.path.join( ft_test, vid ) + '.npy'
    row['seq'] = np.load(ft).shape[0]
    gt = row['gt']
    pt = row['pred']
    # print(row['name'])
    b = bleu( [gt.split()], [pt.split()])
    row['b1']=round( b['b1'], 2 )
    row['b2']=round( b['b2'], 2 )
    row['b3']=round( b['b3'], 2 )
    row['b4']=round( b['b4'], 2 )
    return row

@app.route( '/' )
def home():
    return render_template( 'home.html', headers=headers, res=d )

@app.route( '/process/<vid>' )
def heatmap( vid ):
    # pdb.set_trace()
    filepath = f'{dir}/{filename}'
    if os.path.exists( filepath ):
        os.remove( filepath )
    gt = d[vid]['gt']
    pt = d[vid]['pred']
    value = obj[ vid ]
    df = pd.DataFrame()
    layers = {}
    for f in value.keys():
        for c in value[f].keys():
            for l, val in value[f][c].items():
                if l !='layer3':
                    continue
                val = np.amax( val, axis=1 )
                index = f'{f},{c},{l}'
                if l not in layers.keys():
                    layers[l] = [ index ]
                else:
                    layers[l].append( index )
                temp = pd.DataFrame( data=[val], index=[f'{f},{c},{l}'])
                df = temp if df.size==0 else df.append( temp )

    ft = os.path.join( ft_test, vid ) + '.npy'
    seq = np.load(ft).shape[0]
    df = df.iloc[:, :seq]
    df = df/df.values.max()

    plt.figure( figsize=(32,12) )
    sns.heatmap( df, vmin=0, vmax=1, linewidths=.1, cmap='Spectral' )
    # plt.title( f'f1: Resnet50, f2: Optical Flow, f3: OpenPose \nVideo: {vid}\nGT: {gt}\nPT: {pt}' , fontsize=24 )
    plt.title( f'Video: {vid}\nGT: {gt}\nPT: {pt}' , fontsize=24, pad=10 )
    plt.xlabel('Sequences', fontsize=18, labelpad=10)
    # plt.ylabel('Main feature, Cross modal feature, layer no', fontsize=14, labelpad=10 )

    plt.savefig( f'{filepath}' )
    plt.clf()
    # plt.show()
    return render_template( "home.html", time=time.time())

@app.route( '/send_image/<time>' )
def send_image(time):
    return send_from_directory( dir, filename )

@app.route( '/download_image/<time>', methods=["POST"] )
def download_image(time):
    return send_from_directory( dir, filename, as_attachment=True )


if __name__ == '__main__':
    ############################################################################
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( 'config', help='provide the config file with all arguments' )
    args = parser.parse_args()
   
    ############################################################################
    # configuration
    conf = load_config( args.config )
    test = conf['test']
    sl = conf['sl']
    sep = '|' if sl == 'gsl' else ','
    test_df = pd.read_csv( test, sep=sep )
    if sl == 'asl':
        test_df['Caption'] = test_df['Caption'].apply( lambda x: ' '.join( [ e for e in word_tokenize(x.lower()) if e.isalnum() ] ) )

    csv = conf['csv']
    csv_df = pd.read_csv( csv )
    left_on = conf['left_on']
    right_on = conf['right_on']
    vid_col = conf['vid_col']
    ft_test = conf['resnet50_test']
    pickle = conf['pickle']
    dir = conf['dir']
    filename = 'heatmap.jpg'
    with open( pickle, 'rb' ) as f:
        obj = joblib.load( f )

    ############################################################################
    # prepare
    res = pd.merge( test_df, csv_df, left_on=left_on, right_on=right_on ).drop( left_on, axis=1).drop_duplicates( vid_col )
    res = res.apply( lambda x: score(x, vid_col, ft_test ), axis=1 )
    res = res.set_index( vid_col )
    d = res.to_dict('index')

    headers = ['Video', 'No. of seq', 'Bleu1', 'Bleu2', 'Bleu3', 'Bleu4', 'GT', 'PT']
    app.run( host='0.0.0.0', debug=True )