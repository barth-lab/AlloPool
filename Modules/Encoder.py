import torch
import torch.nn as nn
from torch_geometric.utils import unbatch
from einops import repeat
from Modules.tensor_typing import Float, Int, Bool  # noqa: CODE
from torch_geometric.nn import PointTransformerConv
from Modules.EdgePooling import EdgePool
from Modules.TemporalGCN import AttTGCN
from Modules.SelfAttention import SelfAttention


class EdgePoolEncoder(nn.Module):
    def __init__(
        self,
        seq_len,
        input_dim,
        num_time_steps,
        temporal_in,
        temporal_out,
        pool_in,
        pool_ratio,
        pool_attn_heads,
        pool_depth,
        transformer_depth=2,
        transformer_heads=4,
        position_encoding=None,
        temporal_encoding=True,
        encode_distance=True,
        periods=5,
    ):

        super(EdgePoolEncoder, self).__init__()
        self.encode_distance = encode_distance
        if position_encoding:
            input_dim = input_dim + 64
        self.position_encoding = position_encoding
        self.temporal_encoding = temporal_encoding
        self.seq_len = seq_len

        if encode_distance:
            self.project_distance = PointTransformerConv(
                in_channels=temporal_in,
                out_channels=temporal_in,
                pos_nn=nn.Linear(3, temporal_in),
                attn_nn=None,
            )

            self.attention = SelfAttention(
                temporal_in * 2 if position_encoding else temporal_in,
                temporal_in,
                temporal_in * 4,
                n_head=8,
            )

        if temporal_encoding:
            self.temporal_encoding = nn.Embedding(periods, temporal_in)
        if position_encoding:
            self.position_encoding = nn.Embedding(seq_len, temporal_in)

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, temporal_in, bias=False),
        )

        self.temporal_attn = AttTGCN(
            in_channels=temporal_in,
            out_channels=temporal_out,
            periods=periods,
        )

        self.pooler = EdgePool(
            dim=pool_in,
            depth=pool_depth,
            attn_heads=pool_attn_heads,
            ratio=pool_ratio,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
        )

        self.periods = periods

    def forward(
        self,
        batch,
    ):
        edge_index = batch.persistence_edges
        x = batch.x
        x = torch.stack(unbatch(x, batch=batch.batch), dim=0)  # Shape [T,N,D]

        coors = x[..., :3]
        x = x[..., :13]

        x = self.input_projection(x)

        if self.encode_distance:

            distance_repr = []
            for i in range(self.periods):
                distance_repr.append(
                    self.project_distance(
                        x[i, ...], pos=coors[i, ...], edge_index=edge_index
                    )
                )

            distance_repr = torch.stack(distance_repr, dim=0)
            x = x + distance_repr
            if self.position_encoding:
                embed = self.position_encoding(torch.arange(x.size(1))).to(x.device)
                embed = repeat(embed, "b d -> repeat b d", repeat=x.size(0))
                x = torch.cat([x, embed], dim=-1)

            x = x + self.attention(x)
            all_graph_representation = distance_repr.clone()

        if self.temporal_encoding:
            temporal_embed = torch.arange(self.periods, device=x.device)
            temporal_embed = self.temporal_encoding(temporal_embed)
            temporal_embed = repeat(temporal_embed, "t d -> t n d", n=x.shape[1])
            x = x + temporal_embed

        x = self.temporal_attn(x, edge_index)

        x, edge_index, scores, list_edges, pair_edge = self.pooler(
            x=x, edge_index=edge_index, edge_attr=None
        )

        # x = x + updated_x

        return (
            x,
            edge_index,
            scores,
            list_edges,
            pair_edge,
            x,
            all_graph_representation,
        )
