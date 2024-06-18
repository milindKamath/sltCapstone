from models.mult_model import MULTModel
from models.baseline_joeynmt import Baseline_Joeynmt
from models.mult_joeynmtenc_joeynmtdec import Mult_JoeynmtEnc_JoeynmtDec
from models.baseline_mult import Baseline_Mult
from models.mult_modified import MULTModel_modified
from models.mult_fusion import MULTFusion
from src.loss import XentLoss
import torch

import pdb

################################################################################
# global dict to load correct model
models = {
    'mult': MULTModel,
    'mult_joeynmt': Mult_JoeynmtEnc_JoeynmtDec,
    'baseline_joeynmt': Baseline_Joeynmt,
    'baseline_mult': Baseline_Mult,
    'mult_modified': MULTModel_modified,
    'mult_fusion': MULTFusion
}

################################################################################
# build
def build( conf ):
    ############################################################################
    # build/load model, optimizer, scheduler
    model = build_model( conf )
    optimizer = build_optimizer( conf, model )
    scheduler = build_scheduler( conf, optimizer )

    ############################################################################
    # define loss function
    pad_index = conf['data']['pad_index']
    smoothing = conf['model']['loss']['smoothing']
    translation_loss_func = XentLoss( pad_index=pad_index, smoothing=smoothing )
    if conf['misc']['use_cuda']:
        translation_loss_func.cuda()
    return model, optimizer, scheduler, translation_loss_func

################################################################################
# model
def build_model( conf ):
    func = models.get( conf['model']['type'] )
    model = func( conf )
    return model

################################################################################
# optimzer
def build_optimizer( conf, model ):
    optimizer = conf['model']['optimizer']
    lr = optimizer['lr']
    betas = optimizer['betas']
    eps = optimizer['eps']
    weight_decay = optimizer['weight_decay']
    amsgrad = optimizer['amsgrad'] 

    return torch.optim.Adam( model.parameters(), lr=lr, betas=betas, eps=eps, \
                             weight_decay=weight_decay, amsgrad=amsgrad
                           )

################################################################################
# scheduler
def build_scheduler( conf, optimizer ):
    scheduler = conf['model']['scheduler']
    mode = scheduler['mode']
    patience = scheduler['patience']
    factor = scheduler['factor']
    threshold_mode = scheduler['threshold_mode']
    verbose = scheduler['verbose']

    return torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode=mode, \
                                                       patience=patience, factor=factor, \
                                                       threshold_mode=threshold_mode, verbose=verbose 
                                                     )


