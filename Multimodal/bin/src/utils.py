import os, sys
import yaml
import torch
import joblib
import pdb

def load_config( path ):
    if not os.path.exists( path ):
        print( 'No config file present.' )
        sys.exit(1)
    with open( path, 'r' ) as f:
        conf = yaml.safe_load( f )
    return conf

def save_config( conf ):
    print( f'Saving config file details...' )
    dir = conf['save_path']
    if not os.path.exists( dir ):
        os.mkdir( dir )
    filepath = f'{dir}/config.yaml'
    with open( filepath, 'w' ) as f:
        yaml.dump( conf, f, default_flow_style=False )

def save_model( conf, model, optimizer, scheduler, log, epoch, b4=0.0, best=False ):
    dir = f"{conf['save_path']}/ckpt"
    if not os.path.exists( dir ):
        os.mkdir( dir )
    if best:
        print( 'Saving new best model...' )
        filepath = f'{dir}/best.ckpt'
    else:
        print( 'Saving checkpoint model...' )
        filepath = f"{dir}/{epoch}.ckpt"
    if best == False:
        log = {
            'b4': b4,
            'log': {
                'epoch': epoch,
                'val': {
                    'loss': log['loss'],
                    'bleu': log['bleu']
                }
            }
        }
    obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'log': log
    }
    torch.save( obj, filepath )

def load_checkpoint( ckpt_path, use_cuda=False ):
    print( f'Loading checkpoint {ckpt_path}...' )
    if not os.path.exists( ckpt_path ):
        print( 'Checkpoint not found..' )
        sys.exit(1)
    map_location = 'cuda' if use_cuda else 'cpu'
    checkpoint = torch.load( ckpt_path, map_location=map_location )
    return checkpoint

def save_attn_wts( conf, attn_wts ):
    print( 'Saving attention weights for each video...' )
    obj = attn_wts
    filepath = conf['inference']['save_attn_wts_path']
    with open( filepath, 'wb' ) as f:
        joblib.dump( obj, f, compress=3 )
    
