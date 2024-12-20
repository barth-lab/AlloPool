import torch
import torch.nn as nn 
import torch.nn.functional as F 
import random 
from Modules.GraphTransformer import GraphTransformer
from torch_geometric.nn import PointTransformerConv
from einops import repeat, rearrange
from Modules.GMAN import MSTGCN

class EdgePoolDecoder(nn.Module):

    def __init__(
        self,
        dim,
        depth,
        K,
        nb_chev_filter,
        nb_time_filter,
        num_time_steps,
        len_input,
        time_strides=1,
        edge_dim=16,
        decoder='MSTGCN',
        pred_displ=None,
        propagate_pair_attention=None,
    ):
        
        super(EdgePoolDecoder,self).__init__() 
        self.propagate_pair_attention = propagate_pair_attention
        self.depth = depth 
        self.pred_displ = pred_displ

        self.num_time_steps=num_time_steps
        if decoder == 'MSTGCN':
            self.to_update = MSTGCN(
                nb_block=depth,
                in_channels=dim,
                K=K,
                nb_chev_filter=nb_chev_filter,
                nb_time_filter=nb_time_filter,
                time_strides=time_strides,
                num_for_predict=num_time_steps-1 if pred_displ else num_time_steps,
                len_input=len_input-1 if pred_displ else len_input,
                output_dim=3
            )
        elif decoder=='linear':
            dim = dim -3 
            self.to_update = nn.Sequential(
                nn.Linear(dim,dim*2),
                nn.GELU(),
                nn.Linear(dim*2, dim),
                nn.LayerNorm(dim),
                nn.Linear(dim, 3)
            )

        if self.propagate_pair_attention:
            self.attention_transformer = GraphTransformer(
                dim=128,
                depth=1,
                edge_dim=128,
                heads=2,
                with_feedforwards=None,
                gated_residual=None,
                rel_pos_emb=True
            )

        self.decoder = decoder

        


    def forward(
        self,
        feats,
        edge_index,
        scores,
        input_coords,
        pair_edge,
        temporal_skip,
        all_graph_representation,
        time_variance=None,
        num_time_steps=None,
        pad_backbone=None,
        centered_update = None, 
        pred_all_steps = None,
    ): 

        if self.propagate_pair_attention: 
            feats = self.attention_transformer(
                feats.unsqueeze(0), pair_edge.unsqueeze(0)
            ).squeeze(0)


        if time_variance is not None:
            std = input_coords.std(dim=1, keepdim=True)
            input_coords = input_coords / time_variance
        if self.decoder=='MSTGCN':
            if self.pred_displ is None:
                to_pred_dt = num_time_steps
                num_history = input_coords.shape[0]
                all_frame_representation = all_graph_representation[:num_history,...] # Shape T,N,D
                feats = repeat(feats,'n d -> repeat n d', repeat=num_history) #Shape T,N,D
            else:
                to_pred_dt = num_time_steps - 1
                num_history = input_coords.shape[0] -1
                all_frame_representation = all_graph_representation[:num_history,...]
                feats = repeat(feats, 'n d -> repeat n d', repeat=num_history)
                displacements = input_coords[1:,...] - input_coords[:-1,...]
            features = torch.cat([feats, input_coords],dim=-1).unsqueeze(-1) # Shape [bs,129,T_in]
            features = rearrange(features, 't n d bs -> bs n d t', bs=1, t=num_history)
            pred_coords = self.to_update(features, edge_index) # Ouptut shape [N_res, T_out, 3]

        elif self.decoder=='linear':
            pred_coords = self.to_update(feats).unsqueeze(0)
            

        if time_variance is not None:
            pred_coords = pred_coords * std
            # Need to reconstruct this into coordinates of shape [T_out, N_nodes, 3]
        return feats, pred_coords
            

            