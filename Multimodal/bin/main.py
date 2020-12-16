import os, sys, argparse

from src.utils import load_config, save_config, load_checkpoint
from src.vocabulary import Vocabulary
from src.builders import build, build_optimizer, build_scheduler
from src.training import train

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
    save_config( conf )

    
    ############################################################################
    # build/load model, optimizer, scheduler, translation loss func
    model, optimizer, scheduler, translation_loss_func = build( conf )


    ############################################################################
    # if resume provided
    if conf['training']['resume']:
        print( 'Resume training...' )
        ckpt_path = conf['training']['ckpt']
        checkpoint = load_checkpoint( ckpt_path, conf['misc']['use_cuda'] )
        model.load_state_dict( checkpoint['model'] )
        optimizer.load_state_dict( checkpoint['optimizer'] )
        scheduler.load_state_dict( checkpoint['scheduler'] )
        log = checkpoint['log']
        conf['training']['start_epoch'] = log['log']['epoch'] + 1
        conf['training']['best_log'] = log
        pdb.set_trace()

    ############################################################################
    # transfer learning
    if conf['training']['transfer_learning']:
        print( 'Transfer learning...' )
        ckpt_path = conf['training']['ckpt']
        checkpoint = load_checkpoint( ckpt_path, conf['misc']['use_cuda'] )
        model_dict = model.state_dict()
        m = checkpoint['model']
        new_m = {}
        for k, v in m.items():
            if k.split('.')[0] == 'txt_embed' or k.split('.')[0] == 'decoder' :
                new_m[k] = model_dict[k]
            else:
                new_m[k] = v
        model.load_state_dict( new_m )
        # reset optimizer and schedulder for new model
        optimizer = build_optimizer( conf, model )
        scheduler = build_scheduler( conf, optimizer )
        
    ############################################################################
    # begin training
    params = sum( p.numel() for p in model.parameters() if p.requires_grad )
    print( f'Number of trainable parameters: {params}' )                        # mult_modified: 132896768; 473405440
    
    train( conf, train_loader, val_loader, test_loader, model, optimizer, scheduler, translation_loss_func, vocab_obj )

    sys.exit(0)                    
if __name__ == '__main__':
    main()

