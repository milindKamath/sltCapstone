import torch
from torch import nn
import torch.nn.functional as F

# from modules.transformer import TransformerEncoder
from transformer.transformer_encoder import joey_transformer_encoder
from transformer.transformer_decoder import TransformerDecoder
from transformer.transformer_embeddings import Embeddings, SpatialEmbeddings

import torch.nn.functional as F

import pdb

class Mult_JoeynmtEnc_JoeynmtDec(nn.Module):
    def __init__(self, conf):
        """
        Construct a MulT model.
        """
        super(Mult_JoeynmtEnc_JoeynmtDec, self).__init__()
        self.conf = conf
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
        self.enc_dropout = self.model_params['encoder']['dropout']
        self.enc_num_heads = self.model_params['encoder']['num_heads']
        self.enc_num_layers = self.model_params['encoder']['num_layers']
        self.enc_embed_dropout = self.model_params['encoder']['embed_dropout']
        self.enc_scale = self.model_params['encoder']['scale']
        self.enc_norm_type = self.model_params['encoder']['norm_type']
        self.enc_activation_type = self.model_params['encoder']['activation_type']
        self.enc_ff_size = self.model_params['encoder']['ff_size']

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
        # 1. Spatial embeddings for joeynmt encoder

        self.sgn_embed_f1 = self.spatial_embeddings( self.orig_d_f1 )
        self.sgn_embed_f2 = self.spatial_embeddings( self.orig_d_f2 )
        self.sgn_embed_f3 = self.spatial_embeddings( self.orig_d_f3 )

        ################################################################################
        # 2. Crossmodal Attentions
        if self.f1_only:
            self.trans_f1_with_f2 = self.get_network( self.enc_hidden_d )
            self.trans_f1_with_f3 = self.get_network( self.enc_hidden_d )
            
        if self.f2_only:
            self.trans_f2_with_f1 = self.get_network( self.enc_hidden_d )
            self.trans_f2_with_f3 = self.get_network( self.enc_hidden_d )
           
        if self.f3_only:
            self.trans_f3_with_f1 = self.get_network( self.enc_hidden_d )
            self.trans_f3_with_f2 = self.get_network( self.enc_hidden_d )

        ################################################################################       
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_f1_mem = self.get_network( self.enc_emb )
        self.trans_f2_mem = self.get_network( self.enc_emb )
        self.trans_f3_mem = self.get_network( self.enc_emb )

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

    def spatial_embeddings( self, input_size ):
        return SpatialEmbeddings( embedding_dim=self.enc_hidden_d,
                                  scale=self.enc_scale,
                                  norm_type=self.enc_norm_type,
                                  activation_type=self.enc_activation_type,
                                  num_heads=self.enc_num_heads,
                                  input_size=input_size,
                                )

    def get_network(self, emb_dim ):
        return joey_transformer_encoder( hidden_size=emb_dim,
                                         ff_size=self.enc_ff_size,
                                         num_layers=self.enc_num_layers,
                                         num_heads=self.enc_num_heads,
                                         droput=self.enc_dropout,
                                         emb_dropout=self.enc_embed_dropout
                                       )
            
    def forward(self, f1, f2, f3, txt_input, txt, f1_mask, txt_mask, sgn_lengths, capture_wts=False ):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        # pdb.set_trace()
        ################################################################################       
        # 1. Spatial embeddings for joeynmt encoder
        sgn_mask = f1_mask
        proj_f1 = self.sgn_embed_f1( x=f1, mask=sgn_mask )
        proj_f2 = self.sgn_embed_f2( x=f2, mask=sgn_mask )
        proj_f3 = self.sgn_embed_f3( x=f3, mask=sgn_mask )

        ################################################################################
        # 2. Crossmodal attentions and self attentions

        if self.f1_only:
            h_f1_with_f2 = self.trans_f1_with_f2( proj_f1, src_length=sgn_lengths, mask=sgn_mask, x_k=proj_f2, x_v=proj_f2 )    # Dimension (L, N, d_f1)
            h_f1_with_f3 = self.trans_f1_with_f3( proj_f1, src_length=sgn_lengths, mask=sgn_mask, x_k=proj_f3, x_v=proj_f3 )    # Dimension (L, N, d_f1)
            h_f1 = torch.cat( [ h_f1_with_f2, h_f1_with_f3 ], dim=2 )
            h_f1 = self.trans_f1_mem( h_f1, src_length=sgn_lengths, mask=sgn_mask )   #[seq, batch, 60]
            # if type(h_f1) == tuple:
            #     h_f1 = h_f1[0]
            # last_h_l = last_hs = h_f1[-1]   # Take the last output for prediction

        if self.f2_only:   
            h_f2_with_f1 = self.trans_f2_with_f1( proj_f2, src_length=sgn_lengths, mask=sgn_mask, x_k=proj_f1, x_v=proj_f1 )
            h_f2_with_f3 = self.trans_f2_with_f3 ( proj_f2, src_length=sgn_lengths, mask=sgn_mask, x_k=proj_f3, x_v=proj_f3 )
            h_f2 = torch.cat( [ h_f2_with_f1, h_f2_with_f3 ], dim=2 )
            h_f2 = self.trans_f2_mem( h_f2, src_length=sgn_lengths, mask=sgn_mask )
            # if type( h_f2 ) == tuple:
                # h_f2 = h_f2[0]
            # last_h_a = last_hs = h_f2[-1]

        if self.f3_only:
            h_f3_with_f1 = self.trans_f3_with_f1( proj_f3, src_length=sgn_lengths, mask=sgn_mask, x_k=proj_f1, x_v=proj_f1 )
            h_f3_with_f2 = self.trans_f3_with_f2( proj_f3, src_length=sgn_lengths, mask=sgn_mask, x_k=proj_f2, x_v=proj_f2 )
            h_f3 = torch.cat([h_f3_with_f1, h_f3_with_f2], dim=2)
            h_f3 = self.trans_f3_mem( h_f3, src_length=sgn_lengths, mask=sgn_mask )
            # if type(h_f3) == tuple:
            #     h_f3 = h_f3[0]
            # last_h_v = last_hs = h_f3[-1]
        
        ################################################################################
        # 4. Combined encoder outputs and run through decoder
        
        if self.channels == 3:
            # last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)  # 24,180
            # concatenating output of encoder for translation
            encoder_outputs = torch.cat( [ h_f1, h_f2, h_f3 ], dim=2 )    # gives shape of (seq, b, 180)
            # encoder_outputs = encoder_outputs.permute( 1, 0, 2 )    # to get shape of ( b, seq, 180 ) -> to make it compatible for decoder for joeynmt
        else:
            encoder_outputs = h_f1
    

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
            return encoder_outputs, decoder_outputs, 0
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
                