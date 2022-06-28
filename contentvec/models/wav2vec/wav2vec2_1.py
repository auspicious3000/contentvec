# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    Fp32GroupNorm,
    GroupNormMasked,
    CondLayerNorm,
    MultiheadAttention,
    SamePad,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import index_put
from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer


from fairseq.data.data_utils import lengths_to_padding_mask
class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "group_norm_masked", "layer_norm"}
        self.mode = mode

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            class SequentialMasked(nn.Sequential):
                def forward(self, inputs, mask):
                    inputs = self._modules['0'](inputs)
                    inputs = self._modules['1'](inputs)
                    inputs = self._modules['2'](inputs, mask)
                    inputs = self._modules['3'](inputs)
                    return inputs

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                if mode == "default":
                    return nn.Sequential(
                        make_conv(),
                        nn.Dropout(p=dropout),
                        Fp32GroupNorm(dim, dim, affine=True),
                        nn.GELU(),
                    )
                elif mode == "group_norm_masked":
                    return SequentialMasked(
                        make_conv(),
                        nn.Dropout(p=dropout),
                        GroupNormMasked(dim, dim, affine=True),
                        nn.GELU(),
                )               
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl
            if i == 0:
                self.cl = cl
            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=(mode == "default" or mode == "group_norm_masked") and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x, padding_mask):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                if self.mode == "group_norm_masked":
                    if padding_mask is not None:
                        _, k, stride = self.cl
                        lengths_org = (~padding_mask).long().sum(dim=1)
                        lengths = torch.floor(((lengths_org - k) / stride) + 1).long()
                        padding_mask = (~lengths_to_padding_mask(lengths)).long()
                    x = conv(x, padding_mask)   #padding_mask is numeric
                else:
                    x = conv(x)
            else:
                x = conv(x)

        return x


class TransformerEncoder_1(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )
        
        for _ in range(args.encoder_layers_1):
            self.layers.append(
                TransformerSentenceEncoderLayer_1(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
            )
        
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        if args.encoder_layers_1 > 0:
            self.cond_layer_norm = CondLayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        self.num_layers = args.encoder_layers

        self.apply(init_bert_params)

    def forward(self, x, spk_emb, padding_mask=None, layer=None, tap=False):
        x, layer_results = self.extract_features(x, spk_emb, padding_mask, layer, tap)

        if self.layer_norm_first and layer is None:
            if args.encoder_layers_1 > 0:
                x = self.cond_layer_norm(x, spk_emb)
            else:
                x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, spk_emb, padding_mask=None, tgt_layer=None, tap=False):
        if not self.training and tgt_layer is not None:
            assert tgt_layer >= 0 and tgt_layer < len(self.layers)

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if (not self.training or (dropout_probability > self.layerdrop)) and (i < self.num_layers):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                if tgt_layer is not None or tap:
                    layer_results.append(x.transpose(0, 1))
            if i >= self.num_layers:
                x, z = layer(x, spk_emb, self_attn_padding_mask=padding_mask, need_weights=False)
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer_1(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = CondLayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = CondLayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x, emb)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x, emb)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x, emb)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x, emb)

        return x, attn
