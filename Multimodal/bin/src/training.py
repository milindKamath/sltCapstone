from tqdm import tqdm

from src.metrics import bleu
from src.log import wandb_init, epoch_log
import torch

import pdb

def _step( conf, batch_x, batch_y, batch_masks, sgn_lengths, model, translation_loss_func, set, capture_wts=False ):
    if set == 'train':
        model.train()
    else:
        model.eval()
    ################################################################################
    # obtain values and put them to cuda if use cuda true
    # pdb.set_trace()
    model_type = conf['model']['type']
    if model_type == 'joeynmt' or model_type == 'baseline_mult':
        f1 = batch_x
        if conf['misc']['use_cuda']:
            f1 = f1.cuda() 
    else:
        f1, f2, f3 = batch_x
        if conf['misc']['use_cuda']:
            f1 = f1.cuda()
            f2 = f2.cuda()
            f3 = f3.cuda()

    txt_input, txt, caption = batch_y
    f1_mask, txt_mask = batch_masks
    if conf['misc']['use_cuda']:
        txt_input = txt_input.cuda()
        txt = txt.cuda()
        f1_mask = f1_mask.cuda()
        txt_mask = txt_mask.cuda()

        model = model.cuda()
        
    
    ################################################################################
    # model forward and obtain values
    batch_size = f1.size(0)
    if model_type == 'joeynmt' or model_type == 'baseline_mult':
        encoder_outputs, decoder_outputs = model( f1, \
                                                  txt_input, txt, \
                                                  f1_mask, txt_mask, \
                                                  sgn_lengths
                                                )

    else:
        if capture_wts:
            encoder_outputs, decoder_outputs, attn_wts = model( f1, f2, f3, \
                                                                txt_input, txt, \
                                                                f1_mask, txt_mask, \
                                                                sgn_lengths,
                                                                capture_wts=True
                                                              )
        else:
            encoder_outputs, decoder_outputs = model( f1, f2, f3, \
                                                      txt_input, txt, \
                                                      f1_mask, txt_mask, \
                                                      sgn_lengths
                                                    )

    ################################################################################
    # get loss
    translation_loss, normalized_translation_loss = model.get_translation_loss( decoder_outputs, translation_loss_func, \
                                                                                txt, batch_size 
                                                                              )
    if capture_wts:
        return encoder_outputs, translation_loss, normalized_translation_loss, attn_wts
    else:
        return encoder_outputs, translation_loss, normalized_translation_loss
                                           
                                        

def train_epoch( conf, train_loader, model, optimizer, translation_loss_func ):
    desc = 'Training   ->'
    epoch_translation_loss, epoch_normalized_loss = 0, 0
    for b, ( batch_x, batch_y, batch_masks, sgn_lengths, vid_name ) in tqdm( enumerate( train_loader ), total=len( train_loader ), desc=desc ):
        ################################################################################
        # model forward and obtain outputs
        encoder_outputs, translation_loss, normalized_translation_loss = _step( conf, batch_x, batch_y, batch_masks, sgn_lengths, model, translation_loss_func, set='train' )
        epoch_translation_loss += translation_loss.detach().cpu().numpy()
        epoch_normalized_loss += normalized_translation_loss.detach().cpu().numpy()

        ################################################################################
        # loss backward, optimizer step, reset optimizer gradients
        normalized_translation_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    ################################################################################
    # log values and return it
    log = {
        'loss': {
            'translation': epoch_translation_loss,
            'normalized': epoch_normalized_loss
        }
    }

    return log

################################################################################

def evaluate_epoch( conf, test_loader, model, translation_loss_func, vocab_obj, test=False, capture_wts=False ):
    desc = 'Testing    ->' if test else 'Validation ->'
    epoch_translation_loss, epoch_normalized_loss = 0, 0
    all_gt_sent, all_pred_sent = [], []

    start_index = vocab_obj.vocab_index( '<s>' )
    end_index = vocab_obj.vocab_index( '</s>' )
    max_output_length = conf['misc']['max_output_length']
    
    all_attn_wts = {}
    for b, ( batch_x, batch_y, batch_masks, sgn_lengths, vid_name ) in tqdm( enumerate( test_loader ), total=len( test_loader ), desc=desc ):
        ################################################################################
        # model forward and obtain outputs
        if capture_wts:
            encoder_outputs, translation_loss, normalized_translation_loss, attn_wts = _step( conf, batch_x, batch_y, batch_masks, sgn_lengths, model, translation_loss_func, set='test', capture_wts=capture_wts )
        else:
            encoder_outputs, translation_loss, normalized_translation_loss = _step( conf, batch_x, batch_y, batch_masks, sgn_lengths, model, translation_loss_func, set='test' )
        epoch_translation_loss += translation_loss.detach().cpu().numpy()
        epoch_normalized_loss += normalized_translation_loss.detach().cpu().numpy()

        ################################################################################
        # predict sentences without teacher forcing
        f1_mask, txt_mask = batch_masks
        if conf['misc']['use_cuda']:
            f1_mask = f1_mask.cuda()
        ys = model.pred_decoder( encoder_outputs, f1_mask, start_index, end_index, max_output_length )
        pred_sent = vocab_obj.arrays_to_sentences( ys )

        all_pred_sent += pred_sent
        # gt sent
        txt_input, txt, caption = batch_y
        caption = [ t.split() for t in caption ]
        all_gt_sent += caption

        ################################################################################
        # capture all attn wts with vid name
        if capture_wts:
            # pdb.set_trace()
            for idx in range( len(vid_name) ):
                vid = vid_name[idx]
                temp = {}
                for features, cross_fts in attn_wts.items():
                    temp[features] = {}
                    for ft, layers in cross_fts.items():
                        temp[features][ft] = {}
                        for l, values in layers.items():
                            temp[features][ft][l] = values[idx]
                all_attn_wts[vid] = temp

    ################################################################################
    # get bleu scores
    bleu_scores = bleu( all_gt_sent, all_pred_sent )

    ################################################################################
    # log values and return it
    log = {
        'loss': {
            'translation': epoch_translation_loss,
            'normalized': epoch_normalized_loss
        },
        'bleu': bleu_scores,
        'sent': {
            'gts': all_gt_sent,
            'pts': all_pred_sent
        }
    }
    if capture_wts:
        return log, all_attn_wts
    else:
        return log

################################################################################

def train( conf, train_loader, val_loader, test_loader, model, optimizer, scheduler, translation_loss_func, vocab_obj ):
    start_epoch = conf['training'].get( 'start_epoch', 1 )
    end_epoch = conf['training']['epochs']
    best_log = conf['training'].get( 'best_log', { 'b4': 0.0, 'log': { 'epoch': start_epoch } } )
    best_b4 = best_log['b4']

    # check for learning rate min
    lr_min = conf['training'].get( 'learning_rate_min', 1.0e-8 )

    # for wandb logging
    if conf['misc']['wandb']:
        wandb_init( conf, model )

    for epoch in range( start_epoch, end_epoch+1 ):
        print( f"{'#'*100}")
        print( f'Running epoch: {epoch}' )

        ################################################################################
        # train
        train_log = train_epoch( conf, train_loader, model, optimizer, translation_loss_func )

        ################################################################################
        # val
        val_log = evaluate_epoch( conf, val_loader, model, translation_loss_func, vocab_obj, test=False )

        ################################################################################
        # check b4 improved, then run test set and save model
        test_log = None
        b4 = val_log['bleu']['b4']
        if best_b4 < b4:
            print( f'Found new best model.' )
            # update best log
            best_b4= b4
            # testing log
            test_log = evaluate_epoch( conf, test_loader, model, translation_loss_func, vocab_obj, test=True )

        ################################################################################
        # print performance and logs
        lr = scheduler.optimizer.param_groups[0]["lr"]
        epoch_log( conf,  epoch, lr, model, optimizer, scheduler, train_log, val_log, b4=best_b4, test_log=test_log )

        ################################################################################
        # step scheduler based on b4 score of val
        scheduler.step( b4 )

        ################################################################################
        # check for learning rate min; if less then stop training
        new_lr = scheduler.optimizer.param_groups[0]['lr']
        if new_lr < lr_min:
            print( 'Minimum learning rate reached. Terminating...' )
            break

            
            






        

