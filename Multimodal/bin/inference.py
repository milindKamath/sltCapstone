import os, sys, argparse

from src.utils import load_config, save_config, load_checkpoint, save_attn_wts
from src.vocabulary import Vocabulary
from src.builders import build
from src.training import evaluate_epoch

from data_loaders.dataset import data_loader
import torch

import pdb

def main():
    ############################################################################
    # take the config file path as argument
    parser = argparse.ArgumentParser()
    parser.add_argument( 'config', help='provide the config file with all arguments' )
    args = parser.parse_args()
    if args.config:
        conf = load_config( args.config )
    else:
        print( 'No config file' )
        sys.exit(1)
    ############################################################################

    # set seed
    seed = conf['misc']['seed']
    if conf['misc']['use_cuda']:
        if torch.cuda.is_available():
            torch.cuda.manual_seed( seed )
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print( f'You opted to use cuda but you do not have cuda device available.' )
    else:
        torch.manual_seed( seed )
        torch.set_default_tensor_type('torch.FloatTensor')

    ############################################################################
    # Load vocabulary
    print( f'Loading the vocabulary...' )
    vocab_obj = Vocabulary( conf['data']['vocab'] )
    vocab_size = len( vocab_obj )
    ############################################################################

    ############################################################################
    # Load the dataset
    print( 'Loading the dataset...' )
    train_loader, train_size, dims = data_loader( 'train', conf['data'], conf['model'], vocab_obj )
    val_loader, val_size, _ = data_loader( 'val', conf['data'], conf['model'], vocab_obj )
    test_loader, test_size, _ = data_loader( 'test', conf['data'], conf['model'], vocab_obj )

    ############################################################################
    # add details to configuration and save it
    conf['data']['vocab_size'] = vocab_size
    conf['data']['train_size'] = train_size
    conf['data']['val_size'] = val_size
    conf['data']['test_size'] = test_size
    conf['data']['features']['dims'] = dims
    # save_config( conf )

    ############################################################################
    # build/load model, optimizer, scheduler, translation loss func
    model, optimizer, scheduler, translation_loss_func = build( conf )

    ############################################################################
    # load the model for testing
    ckpt_path = conf['inference']['ckpt']
    checkpoint = load_checkpoint( ckpt_path, conf['misc']['use_cuda'] )
    model.load_state_dict( checkpoint['model'] )
    optimizer.load_state_dict( checkpoint['optimizer'] )
    scheduler.load_state_dict( checkpoint['scheduler'] )

    ############################################################################
    # inference
    capture_wts = conf['model'].get( 'capture_wts', False )
    if capture_wts:
        log, attn_wts = evaluate_epoch( conf, test_loader, model, translation_loss_func, vocab_obj, test=True, capture_wts=True ) 
        ############################################################################
        # log attn_wts
        save_attn_wts( conf, attn_wts )
    else:
        log = evaluate_epoch( conf, test_loader, model, translation_loss_func, vocab_obj, test=True, capture_wts=False ) 

    sys.exit(0)                    
if __name__ == '__main__':
    main()

