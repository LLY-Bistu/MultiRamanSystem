import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_backbone import BaseBackbone
from ..builder import BACKBONES
from .SpatialAttention import SpatialAttention
from .googlenet import GoogLeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    '''
    The positional encoding class is used in the encoder and decoder layers.
    It's role is to inject sequence order information into the data since self-attention
    mechanisms are permuatation equivariant. Naturally, this is not required in the static
    transformer since there is no concept of 'order' in a portfolio.'''

    def __init__(self, window, d_model):
        super().__init__()

        self.register_buffer('d_model', torch.tensor(d_model, dtype=torch.float64))

        pe = torch.zeros(window, d_model)
        for pos in range(window):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))

            for i in range(1, d_model, 2):
                pe[pos, i] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x * torch.sqrt(self.d_model) + self.pe[:, :x.size(1)]


def create_mask(seq_len):
    '''
    Create a mask to be used in the decoder.
    Returns a mask of shape (1, seq_len, seq_len)
    '''
    no_peak_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype('uint8')
    return torch.from_numpy(no_peak_mask)


def get_clones(module, N):
    '''
    This helper function is used to create copies of encoder and decoder layers.
    These copies of encoder/decoder layers are used to construct the
    complete stacked encoder/decoder modules.
    '''
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def scaled_dot_product_attention(k, q, v, mask=None):
    '''
    k : (batch, seq_len_k, heads, d_model)
    q : (batch, seq_len_q, heads, d_model)
    v : (batch, seq_len_v, heads, d_model)

    require seq_len_k == seq_len_v
    '''

    b, _, h, d = k.shape

    k = k.transpose(1, 2).contiguous().view(b * h, -1, d)
    q = q.transpose(1, 2).contiguous().view(b * h, -1, d)
    v = v.transpose(1, 2).contiguous().view(b * h, -1, d)

    scores = torch.matmul(q.float(), k.float().transpose(1, 2))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=2)

    # Scaled dot-product.
    scores = torch.matmul(scores.float(), v.float()).view(b, h, -1, d)
    return scores.transpose(1, 2).contiguous().view(b, -1, h * d)


# 考虑修改多头注意力机制算法
class MultiHeadAttention(nn.Module):
    '''This is a Mult-Head wide self-attention class.'''

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.h = heads
        self.d_model = d_model

        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(d_model, heads * d_model, bias=False)
        self.v_linear = nn.Linear(d_model, heads * d_model, bias=False)
        self.k_linear = nn.Linear(d_model, heads * d_model, bias=False)

        self.unifyheads = nn.Linear(heads * d_model, d_model)

    def forward(self, q, k, v, mask=None):
        b = q.shape[0]

        k = self.k_linear(k.float()).view(b, -1, self.h, self.d_model)
        q = self.q_linear(q.float()).view(b, -1, self.h, self.d_model)
        v = self.v_linear(v.float()).view(b, -1, self.h, self.d_model)

        output = scaled_dot_product_attention(k, q, v, mask=mask)
        output = self.unifyheads(output)

        return output


class FeedForward(nn.Module):
    '''This is a pointwise feedforward network.'''

    def __init__(self, d_model, dff, dropout=0.1):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ELU()
        )

    def forward(self, x):
        x = self.ff(x)
        return x


class EncoderLayer_T(nn.Module):
    '''Encoder layer class.'''

    def __init__(self, heads, d_model, dff, dropout=0.1):
        super().__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dff)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.dropout_1(self.attn(x, x, x))
        x = self.norm_1(x + attn_out)

        ffn_out = self.ff(x)
        x = self.norm_2(x + ffn_out)

        return x


class Encoder_t(nn.Module):
    '''Stacked encoder layers.'''

    def __init__(self, N, pe_window, heads, inp_dim, d_model, dff, dropout):
        super().__init__()

        self.N = N
        self.embedding = nn.Linear(inp_dim, d_model)
        self.pe = PositionalEncoding(pe_window, d_model)
        self.dynamiclayers = get_clones(EncoderLayer_T(heads, d_model, dff, dropout=dropout), N)
        self.embedding_out = nn.Linear(d_model, inp_dim)

    def forward(self, x):
        # x (batch, seq_len, inp_dim)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pe(x)  # (batch, seq_len, d_model)

        for i in range(self.N):
            x = self.dynamiclayers[i](x)  # (batch, seq_len, d_model)
        x = self.embedding_out(x)
        return x  # (batch, seq_len, d_model)


@BACKBONES.register_module()
class Ml4fTransformer(BaseBackbone):
    '''
    Main transformer class.
    experiment : selects sigmoid final activation if 'movement' else linear
    inp_dim_e : number of dimensions of encoder input
    inp_dim_d : number of dimensions of decoder input
    d_model : model embedding dimension
    dff : hidden dimension of feed forward network
    N_e : number of encoder layers
    N_d : number of decoder layers
    heads : number of heads
    '''

    def __init__(self, input_dim=900, num_classes=3, d_model=128, dff=128, N_e=2, heads=4, dropout=0.5):
        super().__init__()

        # assert d_model % heads == 0
        self.input_dim = input_dim
        self.num_classes = num_classes

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=900, out_channels=450, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm1d(450),
        #     nn.ELU(inplace=True),
        # )

        self.encoder = Encoder_t(N_e, self.input_dim, heads, self.input_dim, d_model, dff, dropout=dropout)

    def forward(self, x):
        '''
        x (batch, in_seq_len, inp_dim_e)
        y (batch, tar_seq_len, inp_dim_d)
        '''
        # x = torch.transpose(x, 1, 2)
        # x = self.conv1(x)
        x = x.to(torch.float32)
        # x = torch.transpose(x, 1, 2)
        x = self.encoder(x)
        x = x.squeeze(1)

        return (x,)
