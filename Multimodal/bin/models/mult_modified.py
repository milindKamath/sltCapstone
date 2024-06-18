import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder
from transformer.transformer_decoder import TransformerDecoder
from transformer.transformer_embeddings import Embeddings

import torch.nn.functional as F

import pdb

class MULTModel_modified(nn.Module):
    def __init__(self, conf):
        """
        Construct a MulT model.
        """
        super(MULTModel_modified, self).__init__()
        self.conf = conf
        self.capture_wts = True
        self.vocab_size = self.conf['data']['vocab_size']
        self.pad_index = self.conf['data']['pad_index']
        self.orig_d_f1, self.orig_d_f2, self.orig_d_f3 = self.conf['data']['features']['dims']

        self.model_params = self.conf['model']
        self.f1_only = self.model_params['f1_only']
        self.f2_only = self.model_params['f2_only']
        self.f3_only = self.model_params['f3_only']
        self.channels = self.f1_only + self.f2_only + self.f3_only

        ################################################################################
        # encoder
        self.enc_emb = self.model_params['encoder']['emb_dim']
        self.enc_hidden_d = self.enc_emb//2
        self.enc_attn_dropout_mem = self.model_params['encoder']['attn_dropout_mem']
        self.enc_attn_dropout_f1 = self.model_params['encoder']['attn_dropout_f1']
        self.enc_attn_dropout_f2 = self.model_params['encoder']['attn_dropout_f2']
        self.enc_attn_dropout_f3 = self.model_params['encoder']['attn_dropout_f3']
        self.enc_num_layers = self.model_params['encoder']['num_layers']
        self.enc_num_heads = self.model_params['encoder']['num_heads']
        self.enc_attn_mask = self.model_params['encoder']['attn_mask']
        self.enc_embed_dropout = self.model_params['encoder']['embed_dropout']
        self.enc_relu_dropout = self.model_params['encoder']['relu_dropout']
        self.enc_res_dropout = self.model_params['encoder']['res_dropout']

        ################################################################################
        # decoder
        self.dec_emb = self.enc_emb * self.channels
        self.dec_scale = self.model_params['decoder']['scale']
        self.dec_norm_type = self.model_params['decoder']['norm_type']
        self.dec_activation_type = self.model_params['decoder']['activation_type']
        self.dec_num_layers = self.model_params['decoder']['num_layers']
        self.dec_num_heads = self.model_params['decoder']['num_heads']
        self.dec_max_out_len = self.model_params['decoder']['max_out_len']


        ################################################################################
        # 1. Temporal convolutional layers
        self.proj_f1 = nn.Conv1d( self.orig_d_f1, self.enc_hidden_d, kernel_size=1, padding=0, bias=False )
        self.proj_f2 = nn.Conv1d( self.orig_d_f2, self.enc_hidden_d, kernel_size=1, padding=0, bias=False )
        self.proj_f3 = nn.Conv1d( self.orig_d_f3, self.enc_hidden_d, kernel_size=1, padding=0, bias=False )

        ################################################################################
        # 2. Crossmodal Attentions
        if self.f1_only:
            self.trans_f1_with_f2 = self.get_network( self.enc_hidden_d, self.enc_attn_dropout_f2 )
            self.trans_f1_with_f3 = self.get_network( self.enc_hidden_d, self.enc_attn_dropout_f3 )
            
        if self.f2_only:
            self.trans_f2_with_f1 = self.get_network( self.enc_hidden_d, self.enc_attn_dropout_f1 )
            self.trans_f2_with_f3 = self.get_network( self.enc_hidden_d, self.enc_attn_dropout_f3 )
           
        if self.f3_only:
            self.trans_f3_with_f1 = self.get_network( self.enc_hidden_d, self.enc_attn_dropout_f1 )
            self.trans_f3_with_f2 = self.get_network( self.enc_hidden_d, self.enc_attn_dropout_f2 )

        ################################################################################       
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_f1_mem = self.get_network( self.enc_emb, self.enc_attn_dropout_mem )
        self.trans_f2_mem = self.get_network( self.enc_emb, self.enc_attn_dropout_mem )
        self.trans_f3_mem = self.get_network( self.enc_emb, self.enc_attn_dropout_mem )

        # Projection layers
        self.out_dropout = 0.0
        self.proj1 = nn.Linear(self.dec_emb, self.dec_emb)
        self.proj2 = nn.Linear(self.dec_emb, self.dec_emb)

        ################################################################################
        # 4. Decoder
        self.txt_embed = Embeddings( embedding_dim=self.dec_emb, \
                                     scale=self.dec_scale, \
                                     norm_type= self.dec_norm_type, \
                                     activation_type=self.dec_activation_type, \
                                     vocab_size=self.vocab_size, \
                                     padding_idx=self.pad_index,
                                   )
        self.decoder = TransformerDecoder( vocab_size=self.vocab_size, \
                                           hidden_size=self.dec_emb,
                                           num_layers=self.dec_num_layers,
                                           num_heads= self.dec_num_heads
                                          )

    def get_network(self, emb_dim, attn_dropout ):
        return TransformerEncoder( embed_dim=emb_dim,
                                   num_heads=self.enc_num_heads,
                                   layers=self.enc_num_layers,
                                   attn_dropout=attn_dropout,
                                   relu_dropout=self.enc_relu_dropout,
                                   res_dropout=self.enc_res_dropout,
                                   embed_dropout=self.enc_embed_dropout,
                                   attn_mask=self.enc_attn_mask
                                 )
            
    def forward(self, f1, f2, f3, txt_input, txt, f1_mask, txt_mask, sgn_lengths, capture_wts=False ):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        ################################################################################
        # 1. Temporal Convolutions

        f1 = F.dropout(f1.transpose(1, 2), p=self.enc_embed_dropout, training=self.training)
        f2 = f2.transpose(1, 2)
        f3 = f3.transpose(1, 2)
       
        # Project the textual/visual/audio features
        proj_f1 = f1 if self.orig_d_f1 == self.enc_hidden_d else self.proj_f1( f1 )
        proj_f2 = f2 if self.orig_d_f2 == self.enc_hidden_d else self.proj_f2( f2 )
        proj_f3 = f3 if self.orig_d_f3 == self.enc_hidden_d else self.proj_f3( f3 )

        proj_f2 = proj_f2.permute(2, 0, 1)
        proj_f3 = proj_f3.permute(2, 0, 1)
        proj_f1 = proj_f1.permute(2, 0, 1)

        ################################################################################
        # 2. Crossmodal attentions and self attentions

        if capture_wts:
            attn_wts = {}
        if self.f1_only:
            if capture_wts:
                # pdb.set_trace()
                h_f1_with_f2, attn_wts_f1_f2 = self.trans_f1_with_f2( proj_f1, proj_f2, proj_f2, capture_wts=True )    # Dimension (L, N, d_f1)
                h_f1_with_f3, attn_wts_f1_f3  = self.trans_f1_with_f3( proj_f1, proj_f3, proj_f3, capture_wts=True )    # Dimension (L, N, d_f1)
                attn_wts['f1'] = {
                    'f2': attn_wts_f1_f2,
                    'f3': attn_wts_f1_f3
                }
            else:
                h_f1_with_f2 = self.trans_f1_with_f2( proj_f1, proj_f2, proj_f2 )    # Dimension (L, N, d_f1)
                h_f1_with_f3  = self.trans_f1_with_f3( proj_f1, proj_f3, proj_f3 )    # Dimension (L, N, d_f1)
            h_f1 = torch.cat( [ h_f1_with_f2, h_f1_with_f3 ], dim=2 )
            h_f1 = self.trans_f1_mem( h_f1 )   #[seq, batch, 60]
            # if type(h_f1) == tuple:
            #     h_f1 = h_f1[0]
            # last_h_l = last_hs = h_f1[-1]   # Take the last output for prediction

        if self.f2_only:   
            if capture_wts:
                h_f2_with_f1, attn_wts_f2_f1= self.trans_f2_with_f1( proj_f2, proj_f1, proj_f1, capture_wts=True )
                h_f2_with_f3, attn_wts_f2_f3 = self.trans_f2_with_f3 ( proj_f2, proj_f3, proj_f3, capture_wts=True )
                attn_wts['f2'] = {
                    'f1': attn_wts_f2_f1,
                    'f3': attn_wts_f2_f3
                }
            else:
                h_f2_with_f1 = self.trans_f2_with_f1( proj_f2, proj_f1, proj_f1 )
                h_f2_with_f3 = self.trans_f2_with_f3 ( proj_f2, proj_f3, proj_f3 )
            h_f2 = torch.cat( [ h_f2_with_f1, h_f2_with_f3 ], dim=2 )
            h_f2 = self.trans_f2_mem( h_f2 )
            # if type( h_f2 ) == tuple:
            #     h_f2 = h_f2[0]
            # last_h_a = last_hs = h_f2[-1]

        if self.f3_only:
            if capture_wts:
                h_f3_with_f1, attn_wts_f3_f1 = self.trans_f3_with_f1( proj_f3, proj_f1, proj_f1, capture_wts=True )
                h_f3_with_f2, attn_wts_f3_f2 = self.trans_f3_with_f2( proj_f3, proj_f2, proj_f2, capture_wts=True )
                attn_wts['f3'] = {
                    'f1': attn_wts_f3_f1,
                    'f2': attn_wts_f3_f2
                }
            else:
                h_f3_with_f1 = self.trans_f3_with_f1( proj_f3, proj_f1, proj_f1 )
                h_f3_with_f2 = self.trans_f3_with_f2( proj_f3, proj_f2, proj_f2 )
            h_f3 = torch.cat([h_f3_with_f1, h_f3_with_f2], dim=2)
            h_f3 = self.trans_f3_mem(h_f3)
            # if type(h_f3) == tuple:
            #     h_f3 = h_f3[0]
            # last_h_v = last_hs = h_f3[-1]
        
        ################################################################################
        # 4. Combined encoder outputs and run through decoder

        if self.channels == 3:
            # pdb.set_trace()
            # last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)  # 24,180
            # A residual block
            # last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
            # last_hs_proj += last_hs                 # [ batch, 1536 ]
            # concatenating output of encoder for translation
            encoder_outputs = torch.cat( [ h_f1, h_f2, h_f3 ], dim=2 )    # gives shape of (seq, b, 180)
            encoder_outputs = encoder_outputs.permute( 1, 0, 2 )    # to get shape of ( b, seq, 180 ) -> to make it compatible for decoder for joeynmt
            encoder_outputs_proj = self.proj2(F.dropout(F.relu(self.proj1(encoder_outputs)), p=self.out_dropout, training=self.training))
            encoder_outputs = encoder_outputs_proj + encoder_outputs
        else:
            encoder_outputs = h_f1.permute( 1, 0, 2 )
            encoder_outputs_proj = self.proj2(F.dropout(F.relu(self.proj1(encoder_outputs)), p=self.out_dropout, training=self.training))
            encoder_outputs = encoder_outputs_proj + encoder_outputs
    
        # encoder_outputs = last_hs_proj.unsqueeze(1)
        # decoder
        src_mask = f1_mask
        trg_embed=self.txt_embed( x=txt_input, mask=txt_mask )
        unroll_steps = txt_input.size(1)

        decoder_outputs = self.decoder( encoder_output=encoder_outputs, \
                                        encoder_hidden=None, \
                                        src_mask=src_mask, \
                                        trg_embed=trg_embed, \
                                        trg_mask=txt_mask, \
                                        unroll_steps=unroll_steps, \
                                      )
        if capture_wts:
            return encoder_outputs, decoder_outputs, attn_wts
        else:
            return encoder_outputs, decoder_outputs
    
    def get_translation_loss( self, decoder_outputs, translation_loss_function, txt, batch_size ):
        translation_normalization_mode = self.model_params['loss']['translation_normalization_mode']
        translation_loss_weight = self.model_params['loss']['translation_loss_weight']
        batch_multiplier = self.model_params['loss']['batch_multiplier']

        combined_loss = 0
        word_outputs, _, _, _ = decoder_outputs
        txt_log_probs = F.log_softmax( word_outputs, dim=-1 )
        translation_loss = translation_loss_function( txt_log_probs, txt ) * translation_loss_weight
        if translation_normalization_mode == 'batch':
            txt_normalization_factor = batch_size
            normalized_translation_loss = translation_loss / ( txt_normalization_factor * batch_multiplier )
            combined_loss = normalized_translation_loss
        return translation_loss, combined_loss

    
    # getting decoder outputs for prediction - joeynmt
    def pred_decoder( self, encoder_outputs, f1_mask, start_index, end_index, max_output_length ):
        # resnet50_mask, alexnet_mask, openpose_mask, trg_mask = masks
        src_mask = f1_mask
        batch_size = src_mask.size(0)

        # start with starting index
        ys = encoder_outputs.new_full([batch_size, 1], start_index, dtype=torch.long)

        # a subsequent mask is intersected with this in decoder forward pass
        trg_mask = src_mask.new_ones([1, 1, 1])
        finished = src_mask.new_zeros((batch_size)).byte()

        for _ in range( self.dec_max_out_len ):
            trg_embed = self.txt_embed( ys )

            with torch.no_grad():
                logits, out, _, _ = self.decoder( trg_embed=trg_embed, \
                                                  encoder_output=encoder_outputs, \
                                                  encoder_hidden=None, \
                                                  src_mask=src_mask, \
                                                  unroll_steps=None, \
                                                  hidden=None, \
                                                  trg_mask=trg_mask
                                                )
                logits = logits[:, -1]  # [ b, b, vocab_size ] -> [ b, vocab_size ]
                _, next_word = torch.max(logits, dim=1)
                next_word = next_word.data
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            
            # check if previous symbol was <eos>
            is_eos = torch.eq( next_word, end_index )
            finished += is_eos
            # stop predicting if <eos> reached for all elements in batch
            if (finished >= 1).sum() == batch_size:
                break
        
        # remove start_index token ?
        ys = ys[ :, 1: ]
        return ys.detach().cpu().numpy()
                