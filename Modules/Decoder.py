import torch
import torch.nn as nn
from Modules.GraphTransformer import GraphTransformer
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
        propagate_pair_attention=None,
    ):

        super(EdgePoolDecoder, self).__init__()
        self.propagate_pair_attention = propagate_pair_attention
        self.depth = depth

        self.num_time_steps = num_time_steps
        self.to_update = MSTGCN(
            nb_block=depth,
            in_channels=dim,
            K=K,
            nb_chev_filter=nb_chev_filter,
            nb_time_filter=nb_time_filter,
            time_strides=time_strides,
            num_for_predict=num_time_steps,
            len_input=len_input,
            output_dim=3,
        )

        if self.propagate_pair_attention:
            self.attention_transformer = GraphTransformer(
                dim=128,
                depth=1,
                edge_dim=128,
                heads=2,
                with_feedforwards=None,
                gated_residual=None,
                rel_pos_emb=True,
            )

    def forward(
        self,
        feats,
        edge_index,
        scores,
        input_coords,
        pair_edge,
    ):

        if self.propagate_pair_attention:
            feats = self.attention_transformer(
                feats.unsqueeze(0), pair_edge.unsqueeze(0)
            ).squeeze(0)

        num_history = input_coords.shape[0]
        feats = repeat(feats, "n d -> repeat n d", repeat=num_history)  # Shape T,N,D
        features = torch.cat([feats, input_coords], dim=-1).unsqueeze(
            -1
        )  # Shape [bs,N_res,T_in]
        features = rearrange(features, "t n d bs -> bs n d t", bs=1, t=num_history)
        pred_coords = self.to_update(
            features, edge_index
        )  # Ouptut shape [N_res, T_out, 3]

        return feats, pred_coords
