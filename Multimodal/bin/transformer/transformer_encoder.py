# coding: utf-8

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from signjoey.helpers import freeze_params
from transformer.transformer_layers import TransformerEncoderLayer, PositionalEncoding

import pdb

# pylint: disable=abstract-method
class Encoder(nn.Module):
    """
    Base encoder class
    """

    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size

class joey_transformer_encoder(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        hidden_size: int = 512,             # 512
        ff_size: int = 2048,                # 2048
        num_layers: int = 8,                # 3
        num_heads: int = 4,                 # 8
        dropout: float = 0.1,               # 0.1
        emb_dropout: float = 0.1,           
        freeze: bool = False,
        **kwargs
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(joey_transformer_encoder, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward( self, embed_src: Tensor, src_length: Tensor, mask: Tensor, x_k=None, x_v=None ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        # pdb.set_trace()
        x = self.pe(embed_src)  # add position encoding to word embeddings
        x = self.emb_dropout(x)

        if x_k is not None and x_v is not None:
            # k
            x_k = self.pe( embed_src )
            x_k = self.emb_dropout( x_k )
            # v
            x_v = self.pe( embed_src )
            x_v = self.pe( embed_src )
        
        for layer in self.layers:
            if x_k is not None and x_v is not None:
                x = layer( x, mask, x_k, x_v )
            else:
                x = layer(x, mask)

        return self.layer_norm(x)

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].src_src_att.num_heads,
        )