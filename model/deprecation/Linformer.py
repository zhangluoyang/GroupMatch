"""
Linformer: Self-Attention with Linear Complexity
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 22, 2020
"""
import torch
import torch.nn as nn
from typing import Tuple, Union
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def identity(x, *args, **kwargs):
    return x


def get_act(activation: str):
    if activation == "gelu":
        return F.gelu
    if activation == "relu":
        return F.relu
    return None


def gen_causal_mask(input_size: Tuple[int],
                    dim_k: int,
                    full_attention: bool = False) -> torch.Tensor:
    """
    Generates a causal mask of size (input_size, dim_k) for linformer
    Else, it generates (input_size, input_size) for full attention
    """
    if full_attention:
        return (torch.triu(torch.ones(input_size, input_size)) == 1).transpose(0, 1)
    return (torch.triu(torch.ones(dim_k, input_size)) == 1).transpose(0, 1)


def get_EF(input_size: Tuple[int],
           dim: int,
           method: str = "learnable",
           head_dim: int = None,
           bias: bool = True) -> Union[torch.Tensor, nn.Module]:
    assert method == "learnable" or method == "convolution" \
           or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == "convolution":
        conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size / dim), stride=int(input_size / dim))
        return conv
    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1 / dim)
        return mat
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin


class Residual(nn.Module):
    """
    Implemenation taken from
    https://github.com/lucidrains/sinkhorn-transformer/blob/master/sinkhorn_transformer/sinkhorn_transformer.py
    However, I do postnorm instead of prenorm.
    """

    def __init__(self,
                 fn: nn.Module,
                 input_channels: int = 0,
                 output_channels: int = 0):
        super(Residual, self).__init__()
        self.fn = fn
        self.re_sample = nn.Linear(input_channels, output_channels) if input_channels != output_channels else None
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.re_sample is not None:
            tensor = self.re_sample(tensor) + self.fn(tensor, **kwargs)
            tensor = self.norm(tensor)
            return tensor
        tensor = tensor + self.fn(tensor, **kwargs)
        tensor = self.norm(tensor)
        return tensor


class FeedForward(nn.Module):
    """
    Standard Feed Forward Layer
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 ff_dim: int,
                 dropout: float,
                 activation: str = "gelu"):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(input_channels, ff_dim)
        self.w_2 = nn.Linear(ff_dim, output_channels)
        self.activation = get_act(activation)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        tensor = self.w_1(tensor)
        if self.activation is not None:
            tensor = self.activation(tensor)
        tensor = self.dropout(tensor)
        tensor = self.w_2(tensor)
        tensor = self.dropout2(tensor)
        return tensor


class LinearAttentionHead(nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """

    def __init__(self,
                 dim: int,
                 dropout: float,
                 E_proj: Union[torch.Tensor, nn.Module],
                 F_proj: Union[torch.Tensor, nn.Module],
                 causal_mask: torch.Tensor,
                 full_attention: bool = False):
        super(LinearAttentionHead, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention
        self.causal_mask = causal_mask
        self.is_proj_tensor = isinstance(E_proj, torch.Tensor)

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """
        input_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
        embeddings_mask = kwargs["embeddings_mask"] if "embeddings_mask" in kwargs else None

        # Instead of classic masking, we have to do this, because the classic mask is of size nxn
        if input_mask is not None:
            # This is for k, v
            mask = input_mask[:, :, None]
            K = K.masked_fill_(~mask, 0.0)
            V = V.masked_fill_(~mask, 0.0)
            del mask

        if embeddings_mask is not None:
            mask = embeddings_mask[:, :, None]
            Q = Q.masked_fill_(~mask, 0.0)
            del mask

        K = K.transpose(1, 2)
        if not self.full_attention:
            if self.is_proj_tensor:
                self.E = self.E.to(K.device)
                K = torch.matmul(K, self.E)
            else:
                K = self.E(K)
        Q = torch.matmul(Q, K)

        P_bar = Q / torch.sqrt(torch.tensor(self.dim).type(Q.type())).to(Q.device)
        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(Q.device)
            P_bar = P_bar.masked_fill_(~self.causal_mask, float('-inf'))
        P_bar = P_bar.softmax(dim=-1)

        # Only save this when visualizing
        if "visualize" in kwargs and kwargs["visualize"] == True:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)

        if not self.full_attention:
            V = V.transpose(1, 2)
            if self.is_proj_tensor:
                self.F = self.F.to(V.device)
                V = torch.matmul(V, self.F)
            else:
                V = self.F(V)
            V = V.transpose(1, 2)
        out_tensor = torch.matmul(P_bar, V)
        return out_tensor


class MHAttention(nn.Module):
    """
    Multihead attention, with each head being a Linformer Head
    This feeds directly into a feed forward head
    """

    def __init__(self,
                 input_size: Tuple[int],
                 dim: int,
                 channels: int,
                 dim_k: int,
                 nhead: int,
                 dropout: float,
                 checkpoint_level: str,
                 parameter_sharing: str,
                 E_proj: Union[torch.Tensor, nn.Module],
                 F_proj: Union[torch.Tensor, nn.Module],
                 full_attention: bool,
                 causal_mask: torch.Tensor,
                 w_o_intermediate_dim=None,
                 decoder_mode=False, method="learnable"):
        super(MHAttention, self).__init__()
        self.heads = nn.ModuleList()
        self.input_size = input_size
        self.dim_k = dim_k
        self.channels = channels
        self.causal_mask = causal_mask
        self.checkpoint_level = checkpoint_level
        self.w_o_intermediate_dim = w_o_intermediate_dim
        if parameter_sharing != "layerwise":
            E_proj = get_EF(input_size, dim_k, method, dim)
            F_proj = get_EF(input_size, dim_k, method,
                            dim) if parameter_sharing == "none" or parameter_sharing == "headwise" else E_proj
        self.decoder_mode = decoder_mode
        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()

        for _ in range(nhead):
            # all head share one feature
            if parameter_sharing == "none":
                E_proj = get_EF(input_size, dim_k, method, dim)
                F_proj = get_EF(input_size, dim_k, method, dim)
            attn = LinearAttentionHead(dim, dropout, E_proj, F_proj, causal_mask, full_attention)
            self.heads.append(attn)
            self.to_q.append(nn.Linear(channels, dim, bias=False))
            self.to_k.append(nn.Linear(channels, dim, bias=False))
            self.to_v.append(nn.Linear(channels, dim, bias=False))
        if w_o_intermediate_dim is None:
            self.w_o = nn.Linear(dim * nhead, channels)
        else:
            self.w_o_1 = nn.Linear(dim * nhead, w_o_intermediate_dim)
            self.w_o_2 = nn.Linear(w_o_intermediate_dim, channels)
        self.mh_dropout = nn.Dropout(dropout)

    def forward(self, tensor, **kwargs):
        batch_size, input_len, channels = tensor.shape
        assert not (self.decoder_mode and "embeddings" not in kwargs), "Embeddings must be supplied if decoding"
        assert not ("embeddings" in kwargs and (
            kwargs["embeddings"].shape[0], kwargs["embeddings"].shape[1], kwargs["embeddings"].shape[2]) != (
                        batch_size, input_len, channels)), "Embeddings size must be the same as the input tensor"
        head_outputs = []
        for index, head in enumerate(self.heads):
            Q = self.to_q[index](tensor)
            K = self.to_k[index](tensor) if not self.decoder_mode else self.to_k[index](kwargs["embeddings"])
            V = self.to_v[index](tensor) if not self.decoder_mode else self.to_v[index](kwargs["embeddings"])
            if self.checkpoint_level == "C2":
                head_outputs.append(checkpoint(head, Q, K, V))
            else:
                head_outputs.append(head(Q, K, V, **kwargs))
        out = torch.cat(head_outputs, dim=-1)
        if self.w_o_intermediate_dim is None:
            out = self.w_o(out)
        else:
            out = self.w_o_1(out)
            out = self.w_o_2(out)
        out = self.mh_dropout(out)
        return out


class Linformer(nn.Module):
    """
    My attempt at reproducing the Linformer Paper
    https://arxiv.org/pdf/2006.04768.pdf
    """

    def __init__(self, input_size, channels, dim_k, dim_ff=256, dim_d=None, dropout_ff=0.15, nhead=4, depth=1,
                 dropout=0.1, activation="gelu", checkpoint_level: str = "C0", parameter_sharing="layerwise",
                 k_reduce_by_layer=0, full_attention: bool = False, include_ff=True, w_o_intermediate_dim=None,
                 decoder_mode=False, causal=False, method="learnable", ff_intermediate=None):
        super(Linformer, self).__init__()
        assert activation == "gelu" or activation == "relu", "Only gelu and relu activations supported for now"
        assert checkpoint_level == "C0" or checkpoint_level == "C1" or checkpoint_level == "C2", "Checkpoint level has to be either C0, C1, or C2."
        assert parameter_sharing == "none" or parameter_sharing == "headwise" or parameter_sharing == "kv" or parameter_sharing == "layerwise", "The `parameter_sharing` flag has to be either 'none', 'headwise', 'kv', or 'layerwise'."
        assert channels % nhead == 0 if dim_d is None else True, "If `dim_d` is not set to a custom value, `channels` must be divisible by `nhead`!"
        assert not (
                ff_intermediate and parameter_sharing == "layerwise"), "Parameter sharing must not be layerwise if ff_intermediate is enabled!"
        assert not (
                ff_intermediate and decoder_mode), "Raising the dimension in the middle cannot be done in the decoder!"

        layers = nn.ModuleList()
        self.decoder_mode = decoder_mode
        self.input_size = input_size
        self.channels = channels
        self.checkpoint_level = checkpoint_level
        self.depth = depth
        self.nhead = nhead

        head_dim = channels // nhead if dim_d is None else dim_d

        E_proj = get_EF(input_size, dim_k, method, head_dim)
        causal_mask = gen_causal_mask(input_size, dim_k, full_attention) if causal else None
        # If we want causal but only with the encoder
        causal_enc = gen_causal_mask(input_size, dim_k, full_attention) if (causal and not decoder_mode) else None

        get_attn = lambda attn_channels, curr_dim_k: MHAttention(input_size, head_dim, attn_channels, curr_dim_k, nhead,
                                                                 dropout, checkpoint_level, parameter_sharing, E_proj,
                                                                 E_proj, full_attention, causal_enc,
                                                                 w_o_intermediate_dim, decoder_mode=False,
                                                                 method=method)
        get_attn_context = lambda attn_channels, curr_dim_k: MHAttention(input_size, head_dim, attn_channels,
                                                                         curr_dim_k, nhead, dropout, checkpoint_level,
                                                                         parameter_sharing, E_proj, E_proj,
                                                                         full_attention, causal_mask,
                                                                         w_o_intermediate_dim, decoder_mode=True,
                                                                         method=method)
        get_ff = lambda input_channels, output_channels: FeedForward(input_channels, output_channels, dim_ff,
                                                                     dropout_ff, activation)

        for index in range(depth):
            input_channels = ff_intermediate if (
                                                        index != 0 and ff_intermediate is not None) \
                                                and not decoder_mode else channels
            output_channels = ff_intermediate if (
                                                         index != depth - 1 and ff_intermediate is not None) \
                                                 and not decoder_mode else channels
            # TODO: Change the input and output channels here
            attn_layer = get_attn(input_channels, max(1, dim_k - index * k_reduce_by_layer))
            ff_layer = get_ff(input_channels, output_channels)

            attn_layer, ff_layer = map(lambda res_ch_in, res_ch_out, fn: Residual(fn, res_ch_in, res_ch_out),
                                       (input_channels, input_channels), (input_channels, output_channels),
                                       (attn_layer, ff_layer))

            if include_ff:
                layers.extend([attn_layer, ff_layer])
            else:
                layers.extend([attn_layer])

            if not self.decoder_mode:
                continue

            attn_context = get_attn_context(channels, max(1, dim_k - index * k_reduce_by_layer))
            ff_context = get_ff(channels, channels)

            attn_context, ff_context = map(lambda fn: Residual(fn, channels, channels), (attn_context, ff_context))

            if include_ff:
                layers.extend([attn_context, ff_context])
            else:
                layers.extend([attn_context])

        self.seq = layers

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len, channels)
        """
        bt, n, c = tensor.shape
        assert n == self.input_size, "This tensor is of the wrong size. Dimension 1 has to match the `input_size` flag"
        assert c == self.channels, "This tensor is of the wrong size. Dimension 2 has to match the `channels` flag"
        assert self.checkpoint_level == "C0" if kwargs else True, "Cannot run checkpointing when using kwargs. Please set the checkpoint level to `C0`"
        assert "embeddings" not in kwargs or self.decoder_mode, "If decoding, needs to be initialized with `decoder_mode=True`"

        for layer in self.seq:
            if self.checkpoint_level != "C0":
                tensor = checkpoint(layer, tensor)
            else:
                tensor = layer(tensor, **kwargs)
        return tensor
