import os, sys
import torch
import wandb as wd
from src.utils import save_model

import pdb

def print_performance( loss, bleu=None, lr=None, set='train' ):
    # pdb.set_trace()
    if set == 'train':
        print(
            f'\n{set:6}-> ' \
            f"translation loss: {loss['translation']:3.5f} | normalized loss: {loss['normalized']:3.5f} | " \
            f"lr: {lr}"
        )
    else:
        print(
            f'{set:6}-> ' \
            f"translation loss: {loss['translation']:3.5f} | normalized loss: {loss['normalized']:3.5f} | " \
            f"bleu1: {bleu['b1']:3.5f} | bleu2: {bleu['b2']:3.5f} | bleu3: {bleu['b3']:3.5f} | bleu4: {bleu['b4']:3.5f}"
        )

def log_best( conf, log, sent ):
    print( 'Logging best model results...' )
    train = log['log']['train']
    val = log['log']['val']
    test = log['log']['test']


    dir = f"{conf['save_path']}/log"
    if not os.path.exists( dir ):
        os.mkdir( dir )
    filepath = f'{dir}/best.txt'
    s = f"Best results:\n\n"
    s += f"Best score: {log['b4']:3.5f}\n"
    s += f"Epoch: {log['log']['epoch']}\n"
    s += f"Train   -> translation loss: {train['loss']['translation']:3.5f} | normalized loss: {train['loss']['normalized']:3.5f}\n"
    s += f"Valid   -> translation loss: {val['loss']['translation']:3.5f} | normalized loss: {val['loss']['normalized']:3.5f} | "
    s += f"bleu1: {val['bleu']['b1']:3.5f} | bleu2: {val['bleu']['b2']:3.5f} | bleu3: {val['bleu']['b3']:3.5f} | bleu4: {val['bleu']['b4']:3.5f}\n"
    s += f"Test    -> translation loss: {test['loss']['translation']:3.5f} | normalized loss: {test['loss']['normalized']:3.5f} | "
    s += f"bleu1: {test['bleu']['b1']:3.5f} | bleu2: {test['bleu']['b2']:3.5f} | bleu3: {test['bleu']['b3']:3.5f} | bleu4: {test['bleu']['b4']:3.5f}\n"

    s += '\nTest Sentences:\n'
    gts, pts = sent['gts'], sent['pts']
    for i in range( len(gts) ):
        gt = ' '.join( gts[i] )
        pt = ' '.join( pts[i] )
        s += f"GT_{i+1}: { gt }\n"
        s += f"PT_{i+1}: { pt }\n"

    with open( filepath, 'w' ) as f:
        f.write(s)

def log_performance( conf, epoch, lr, train_log, val_log, test_log=None ):
    print( 'Logging epoch performance...' )
    dir = f"{conf['save_path']}/log"
    if not os.path.exists( dir ):
        os.mkdir( dir )

    filepath = f'{dir}/performance.txt'
    s = ''
    if not os.path.exists( filepath ):
        s += 'epoch,lr,train_translation_loss,train_normalized_loss,val_translation_loss,val_normalized_loss,test_translation_loss,test_normalized_loss'
        s += ',test_bleu1,test_bleu2,test_bleu3,test_bleu4,val_bleu1,val_bleu2,val_bleu3,val_bleu4,\n'
    with open( filepath, 'a' ) as f:
        s += f"{epoch},{lr},{train_log['loss']['translation']:3.5f},{train_log['loss']['normalized']:3.5f},{val_log['loss']['translation']:3.5f},{val_log['loss']['normalized']:3.5f}"
        if test_log:
            test_loss = test_log['loss']
            test_bleu = test_log['bleu']
            s += f",{test_loss['translation']:3.5f},{test_loss['normalized']:3.5f}"
            s += f",{test_bleu['b1']:3.5f},{test_bleu['b2']:3.5f},{test_bleu['b3']:3.5f},{test_bleu['b4']:3.5f}"
        else:
            s += f",-1,-1,-1,-1,-1,-1"
        s += f",{val_log['bleu']['b1']:3.5f},{val_log['bleu']['b2']:3.5f},{val_log['bleu']['b3']:3.5f},{val_log['bleu']['b4']:3.5f}\n"
        f.write( s )

def write_ckpt( conf, epoch, val_sent):
    print( 'Writing checkoint sentences...' )
    dir = f"{conf['save_path']}/log"
    if not os.path.exists( dir ):
        os.mkdir( dir )
    filepath = f'{dir}/ckpt.txt'
    s = f"{'#'*20}"
    s += f'\nEpoch {epoch}\n\n'
    gts, pts = val_sent['gts'], val_sent['pts']
    for i in range( len(gts) ):
        gt = ' '.join( gts[i] )
        pt = ' '.join( pts[i] )
        s += f"GT_{i+1}: { gt }\n"
        s += f"PT_{i+1}: { pt }\n"
    s += f"\n{'#'*20}\n\n"

    with open( filepath, 'a' ) as f:
        f.write(s)

def wandb_model( epoch, lr, commit=False ):
    wd.log(
        {
            'epoch': epoch,
            'lr': lr
        },
        commit=commit
    )

def wandb_loss( loss, set='train', commit=False ):
    wd.log(
        {
            f'{set}_translation_loss': loss['translation'],
            f'{set}_normalized_loss': loss['normalized'],
        },
        commit=commit
    )

def wandb_bleu( bleu, set='train', commit=False ):
    wd.log(
        {
            f'{set}_bleu1': bleu['b1'],
            f'{set}_bleu2': bleu['b2'],
            f'{set}_bleu3': bleu['b3'],
            f'{set}_bleu4': bleu['b4'],
        },
        commit=commit
    )

# sentences
def wandb_sent( epoch, sent, set='val', commit=False ): 
    gts, pts = sent['gts'], sent['pts']
    table = wd.Table( columns=[ 'epoch', 'gt', 'pred' ] )
    for i in range( len(gts) ):
        gt = ' '.join( gts[i] )
        pt = ' '.join( pts[i] )
        table.add_data( epoch, gt, pt )
    wd.log(
        {
            f'{set} sentences': table
        },
        commit = commit
    )

def epoch_log( conf, epoch, lr, model, optimizer, scheduler, train_log, val_log, b4=0.0, test_log=None ):
    # print performances
    print_performance( train_log['loss'], lr=lr, set='train' )
    print_performance( val_log['loss'], bleu=val_log['bleu'], set='valid' )

    ################################################################################
    # if test log, then best model found. log and save it
    if test_log:
        print_performance( test_log['loss'], bleu=test_log['bleu'], set='test' )
        best_log = {
            'b4': val_log['bleu']['b4'],
            'log': {
                'epoch': epoch,
                'train': {
                    'loss': train_log['loss']
                },
                'val': {
                    'loss': val_log['loss'],
                    'bleu': val_log['bleu']
                },
                'test': {
                    'loss': test_log['loss'],
                    'bleu': test_log['bleu']
                }
            }
        }
        # log best and save best model
        log_best( conf, best_log, test_log['sent'] )
        save_model( conf, model, optimizer, scheduler, best_log, epoch, best=True )

    ################################################################################
    # log performance
    log_performance( conf, epoch, lr, train_log, val_log, test_log=test_log )

    ################################################################################
    # log and save model after ckpt
    if epoch % conf['training']['save_model_epoch'] == 0:
        save_model( conf, model, optimizer, scheduler, val_log, epoch, b4=b4 )
    
    if epoch % conf['training']['save_sent_epoch'] == 0:
        write_ckpt( conf, epoch, val_log['sent'] )

    ################################################################################
    # log wandb
    if conf['misc']['wandb']:
        if test_log:
            wandb_loss( test_log['loss'], set='test' )
            wandb_bleu( test_log['bleu'], set='test' )
            wandb_sent( epoch, test_log['sent'], set='test' )
        
        if epoch % conf['training']['save_sent_epoch'] == 0:
            wandb_sent( epoch, val_log['sent'], set='val' )

        # other logs
        # log
        wandb_model( epoch, lr )
        wandb_loss( train_log['loss'], set='train' )
        wandb_loss( val_log['loss'], set='val' )
        wandb_bleu( val_log['bleu'], set='val', commit=True )

    return


def wandb_init( conf, model ):
    wd.init( name=f"Expt {conf['expt']}", notes=conf['notes'], config=conf, project="multimodal", dir=conf['save_path'], save_code=True )
    wd.watch( model )



    
