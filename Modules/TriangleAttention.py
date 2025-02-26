import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import einsum
from einops.layers.torch import Rearrange
from inspect import isfunction


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class TriangleMultiplication(nn.Module):
    """
    First module of the pairwise triangular attention.
    Two different steps are performed in this module:
        1. Triangle multiplicative update using outgoing edges
        2. Triangle multiplicative update using incoming edges

    This will iteratively updates edge value based on the other edge values. Can be seen as a specific message passing function for edges

    Input:

    """

    def __init__(self, dim, mix, hidden_dim=None):

        super(TriangleMultiplication, self).__init__()
        assert mix in {
            "incoming",
            "outgoing",
        }, " Mix must be either incoming or outgoing"
        hidden_dim = dim
        self.norm = nn.LayerNorm(dim)

        # Perform here the linear for left projection and right projection
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # Initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 1.0)

        if mix == "outgoing":
            self.mix_einsum = "... i k d, ... j k d -> ... i j d"
        elif mix == "incoming":
            self.mix_einsum = "... k j d, ... k i d -> ... i j d"

        self.to_out_norm = nn.LayerNorm(hidden_dim)

        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        # X shape is [Nres, Nres, dim]

        assert x.shape[0] == x.shape[1]

        if mask is not None:
            mask = rearrange(mask, "i j -> i j ()")

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if mask is not None:
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum, left, right)

        out = self.to_out_norm(out)
        out = self.to_out(out)
        out = out * out_gate
        return out


class Attention(nn.Module):
    def __init__(
        self, dim, seq_len=None, heads=4, dim_head=64, dropout=0.0, gating=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.0)
        nn.init.constant_(self.gating.bias, 1.0)

        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(
        self,
        x,
        mask=None,
        attn_bias=None,
        context=None,
        context_mask=None,
        tie_dim=None,
    ):
        device, _, h, has_context = (
            x.device,
            x.shape,
            self.heads,
            exists(context),
        )

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        i, _ = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # scale

        q = q * self.scale

        # query / key similarities

        if exists(tie_dim):
            # as in the paper, for the extra MSAs
            # they average the queries along the rows of the MSAs
            # they named this particular module MSAColumnGlobalAttention

            q, k = map(
                lambda t: rearrange(t, "(b r) ... -> b r ...", r=tie_dim), (q, k)
            )
            q = q.mean(dim=1)

            dots = einsum("b h i d, b r h j d -> b r h i j", q, k)
            dots = rearrange(dots, "b r ... -> (b r) ...")
        else:
            dots = einsum("b h i d, b h j d -> b h i j", q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)

        if exists(attn_bias):
            dots = dots + attn_bias

        # masking

        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device=device).bool())
            context_mask = (
                mask
                if not has_context
                else default(
                    context_mask,
                    lambda: torch.ones(1, k.shape[-2], device=device).bool(),
                )
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        # gating

        gates = self.gating(x)
        out = out * gates.sigmoid()

        # combine to out

        out = self.to_out(out)
        return out


class AxialAttention(nn.Module):
    """Axial attention for column or row attention from a pairwise representation"""

    def __init__(
        self,
        dim,
        heads,
        row_attn=True,
        col_attn=True,
        accept_edges=False,
        global_query_attn=False,
        **kwargs
    ):

        super(AxialAttention, self).__init__()

        assert not (
            not row_attn and not col_attn
        ), "row or column attention must be turned of, cannot do on both at the same time"

        self.row_attn = row_attn
        self.col_attn = col_attn

        self.global_query_attn = global_query_attn

        self.norm = nn.LayerNorm(dim)

        self.attn = Attention(dim=dim, heads=heads, **kwargs)

        if accept_edges == True:
            self.edges_to_attn_bias = nn.Sequential(
                nn.Linear(dim, heads, bias=False), Rearrange("n1 n2 h -> h n1 n2")
            )

    def forward(self, x, edges=None, mask=None):
        assert self.row_attn ^ self.col_attn

        (
            n1,
            n2,
            h,
        ) = x.shape  # Input x is a pairwise representation matrix of shape [N, N, dim]

        x = self.norm(x)

        # Axial attention
        if self.col_attn:
            axial_dim = n2
            mask_fold_axial = "n1 n2 -> n2 n1"
            input_fold = "n1 n2 h -> n2 n1 h"
            out_fold = "n2 n1 h -> n1 n2 h"

        if self.row_attn:
            axial_dim = n1
            mask_fold_axial = "n1 n2 -> n1 n2"
            input_fold = "n1 n2 h -> n1 n2 h"
            out_fold = "n1 n2 h -> n1 n2 h"

        x = rearrange(x, input_fold)

        if mask is not None:
            mask = rearrange(mask, mask_fold_axial)

        attn_bias = None

        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, "h i j -> x h i j", x=axial_dim)

        tie_dim = axial_dim if self.global_query_attn else None

        out = self.attn(x, mask=mask, attn_bias=attn_bias, tie_dim=tie_dim)
        out = rearrange(out, out_fold, n1=n1, n2=n2)

        return out


class OuterMean(nn.Module):
    def __init__(self, dim, hidden_dim=None, eps=1e-5):

        super(OuterMean, self).__init__()

        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        hidden_dim = dim

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, "i d -> i () d") * rearrange(right, "j d -> () j d")

        if exists(mask):
            # masked mean, if there are padding in the rows of the MSA
            mask = rearrange(mask, "m i -> m i () ()") * rearrange(
                mask, "m j -> m () j ()"
            )
            outer = outer.masked_fill(~mask, 0.0)
            outer = outer.mean(dim=1) / (mask.sum(dim=1) + self.eps)
        else:
            outer = outer.mean(dim=1)

        return self.proj_out(outer)


class PairwiseAttention(nn.Module):
    """ """

    def __init__(
        self, dim, heads, dim_head, dropout=0.1, global_column_attn=False, **kwargs
    ):

        super(PairwiseAttention, self).__init__()

        self.outer_mean = OuterMean(dim)

        self.triangle_outgoing = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            row_attn=True,
            col_attn=False,
            accept_edges=True,
            global_query_attn=global_column_attn,
        )

        self.triangle_incoming = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            row_attn=False,
            col_attn=True,
            accept_edges=True,
            global_query_attn=global_column_attn,
        )

        self.triangle_mult_outoing = TriangleMultiplication(dim=dim, mix="outgoing")
        self.triangle_mult_incoming = TriangleMultiplication(dim=dim, mix="incoming")

    def forward(self, x, mask=None, seq_repr=None):
        """
        Apply the triangle representation on x.
        Inputs:
            - x: n_res * n_res * dim

        """
        if seq_repr is not None:
            x = x + self.outer_mean(seq_repr, mask=mask)

        # print('X start')
        x = self.triangle_mult_outoing(x, mask=mask) + x
        x = self.triangle_mult_incoming(x, mask=mask) + x
        x = self.triangle_outgoing(x, edges=x, mask=mask) + x
        x = self.triangle_incoming(x, edges=x, mask=mask) + x
        return x
