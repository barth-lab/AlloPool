from torch_geometric.nn import GATv2Conv, TransformerConv

import torch 
import torch.nn as nn 
from einops import rearrange 
from torch import einsum

import random
from Modules.tensor_typing import Float, Int, Bool 
from Modules.TriangleAttention import PairwiseAttention

from torch_geometric.nn.pool.select import SelectTopK


class EdgePoolBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        ratio = 0.5,
        min_score = None,
        multiplier =1,
        dropout=None,
        skip_connexion = None,
        skip=None,
        propagator='GAT'
    ): 
        
        super(EdgePoolBlock,self).__init__()
        self.multiplier = multiplier
        self.skip_connexion = skip_connexion
        self.min_score = min_score
        self.ratio = ratio 
        self.skip = skip

        if propagator=='transformer':
            self.propagate_to_graph = TransformerConv(
                in_channels = in_channels,
                out_channels= in_channels,
                heads=heads,
                edge_dim=1,
                concat=False
            )
        elif propagator=='GAT':
            self.propagate_to_graph = GATv2Conv(
                in_channels=in_channels,
                out_channels=in_channels,
                heads=heads,
                edge_dim=1,
                concat=False,
            )


        self.edge_attention = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels*4),
            nn.ReLU(),
            nn.Linear(in_channels*4, in_channels*2)
        )
        self.pair_attn = PairwiseAttention(
            dim=in_channels,
            heads=2,
            dim_head=64
        )
        self.pair_to_score = nn.Sequential(
            #nn.LayerNorm(in_channels),
            nn.Linear(in_channels,1), 
            nn.Sigmoid(),
        )

        self.select = SelectTopK(1, ratio, min_score, 'tanh')

    def forward(
        self,
        x: Float['n d'],
        edge_index: Int['2 n'],
        edge_attr: Float['2 n'] = None,
        pair_edge: Float['n n d'] = None,
        pair_attn: Float['n n d'] = None,
        edge_index_perm = None,
        recycle_pair_attention = True,
    ):        
        if self.skip: 
            x_skip = x 

        if pair_attn is None: 
            ingoing,outgoing = self.edge_attention(x).chunk(2,dim=-1)
            pair_attn = rearrange(ingoing,'b i -> b () i') + rearrange(outgoing, 'b i -> () b i') #Shape [N_nodes, N_nodes, dim]

        # Create mask for pair_attention only on existing edges
        mask = torch.zeros((x.shape[0], x.shape[0]))
        mask[edge_index[0], edge_index[1]]=1
        mask = mask.bool().to(x.device)
        
        pair_attn = self.pair_attn(pair_attn, seq_repr=None, mask=mask)  # [N,N,dim]

        if recycle_pair_attention:
            if pair_edge is None:
                pair_edge = torch.zeros(pair_attn.shape[0],pair_attn.shape[0],1, device=pair_attn.device)
                pair_edge = pair_edge + self.pair_to_score(pair_attn)
            else:
                pair_edge = pair_edge + self.pair_to_score(pair_attn)
        else:
            pair_edge = self.pair_to_score(pair_attn) # Shape [N, N, dim]
            print('pair_edge', pair_edge)
        filtered_pair_edges = pair_edge[mask] #Shape [N_edge, 1]

        # Compute score for each edge

        select_out = self.select(filtered_pair_edges)
        values = select_out.weight
        perm = select_out.node_index
        edge_index = edge_index[:,perm]# * score.view(-1,1)
        
        # Propagate pooling information to the original graph 
        x = x + self.propagate_to_graph(x, edge_index, edge_attr=values)
        x = self.multiplier * x #if self.multiplier != 1 else x

        if self.skip:
            x = x + x_skip

        return x, edge_index, values, perm, values, pair_edge, pair_attn

        
class EdgePool(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        attn_heads,
        ratio,
        transformer_depth=2,
        transformer_heads=4,
    ):
        
        super(EdgePool,self).__init__() 

        self.depth = depth
        self.pool_layer = nn.ModuleList([])
    
        for i in range(depth):
            self.pool_layer.append(
                EdgePoolBlock(
                    in_channels=dim,
                    heads=attn_heads,
                    ratio=ratio,
                )
            )

        self.post_pool_transition = nn.ModuleList([])

        for i in range(transformer_depth):
            self.post_pool_transition.append(
                GATv2Conv(
                    in_channels=dim,
                    out_channels=dim,
                    heads=transformer_heads,
                    concat=False,
                    edge_dim=1
                )
            )

    def forward(
        self,
        x: Float['n d'],
        edge_index: Int['2 n'],
        edge_attr: Float['n 1'],
    ):
        
        #Perform edge pooling on incoming graph
        list_edges = [] #To retrieve edge indices after pooling 
        scores = torch.ones(self.depth, edge_index.size(1)) * -torch.inf # To retrieve scores after
        scores = scores.to(x.device)


        edge_map = torch.arange(0, edge_index.size(1)).to(edge_index.device)

        pair_attn=None
        pair_edge=None
        out_pooling = x
        for i,pooler in enumerate(self.pool_layer):
            out_pooling, edge_index, edge_attr, edge_map, values, pair_edge, pair_attn = pooler( # Try put back x cause was out_pooling
                x=out_pooling, edge_index=edge_index,edge_attr=edge_attr,edge_index_perm=edge_map, pair_attn=pair_attn, pair_edge=pair_edge
            )

            list_edges.append(edge_map)
            scores[i,edge_map] = values.squeeze(-1)
        x = x + out_pooling
        return x, edge_index, scores, list_edges, pair_attn








