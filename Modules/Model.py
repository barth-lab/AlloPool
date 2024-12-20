import torch 
import torch.nn as nn
import torch.nn.functional as F 
from Modules.Encoder import EdgePoolEncoder
from Modules.Decoder import EdgePoolDecoder
from torch_geometric.utils import unbatch
import random 
import einx

class EdgePoolModel(nn.Module):

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
        transformer_depth,
        transformer_K,
        transformer_nb_cheb_filter,
        transformer_time_cheb_filter,
        len_input,
    ): 
        self.num_time_steps = num_time_steps
        super(EdgePoolModel,self).__init__() 

        self.encoder = EdgePoolEncoder(
            seq_len=seq_len,
            input_dim=input_dim,
            num_time_steps=num_time_steps,
            temporal_in=temporal_in,
            temporal_out=temporal_out,
            pool_in=pool_in,
            pool_ratio=pool_ratio,
            pool_depth=pool_depth,
            pool_attn_heads=pool_attn_heads,
        )

        self.decoder = EdgePoolDecoder(
            depth=transformer_depth,
            dim=temporal_out+3,
            K=transformer_K,
            nb_chev_filter=transformer_nb_cheb_filter,
            nb_time_filter=transformer_time_cheb_filter,
            num_time_steps=num_time_steps,
            len_input=len_input,
        )

    def encode(
        self,
        batch,
        pred_all=None,
    ):
        x,edge_index,scores,list_edges, pair_edge, temporal_skip, all_graph_representation = self.encoder(batch)
        x_time  = torch.stack(unbatch(batch.x, batch=batch.batch), axis=0)#[:-1] # T, N, d
        if pred_all:
            input_coords = x_time[0:1,:,:3] #Input coordinates
            true_coords = x_time[:,:,:3]
        else:
            to_pred_dt = x_time.shape[0] - self.num_time_steps
            input_coords = x_time[0:to_pred_dt,:,:3]
            true_coords = x_time[to_pred_dt:,:,:3]

        return x, edge_index, scores, list_edges,pair_edge, input_coords, true_coords, temporal_skip, all_graph_representation

    def decode(
        self,
        x,
        edge_index,
        scores,
        list_edge,
        pair_edge,
        input_coords,
        true_coords,
        temporal_skip,
        all_graph_representation,
        pred_displ=None,
    ): 

        features, pred_coords = self.decoder(
            feats=x,
            edge_index=edge_index,
            scores=scores,
            input_coords = input_coords,
            pair_edge=pair_edge,
            temporal_skip=temporal_skip,
            all_graph_representation=all_graph_representation,
            num_time_steps=self.num_time_steps,
        )
        #Add calculate loss here 
        all_loss = {}
        if pred_displ:
            true_coords = true_coords[1:,...]
        rmsd_loss, all_rmsd = self.kabsch_rmsd(pred_coords,true_coords)
        translation_loss = self.translation_loss(pred_coords, true_coords)
        contact_loss = self.contact_loss(pred_coords, true_coords)
        centered_loss = self.centered_loss(pred_coords)
        CA_bond_loss = self.CA_distance_loss(pred_coords, true_coords)
        all_loss["rmsd_loss"] = rmsd_loss
        all_loss["translation_loss"] = translation_loss
        all_loss["centered_loss"] = centered_loss
        all_loss["CA_bond_loss"] = CA_bond_loss
        all_loss["contact_loss"] = contact_loss
        for i in range(len(all_rmsd)): 
            all_loss[f"rmsd_state_{i}"] = all_rmsd[i]

        return all_loss, pred_coords, edge_index, scores, pair_edge

    def forward(
        self,
        batch
    ):
        x, edge_index, scores, list_edges,pair_edge, input_coords, true_coords,temporal_skip, all_graph_representation = self.encode(batch)
        return self.decode(x, edge_index, scores, list_edges, pair_edge, input_coords, true_coords, temporal_skip, all_graph_representation)
        

    def find_alignment(
        self,
        P,
        Q
    ):
        centroid_P, centroid_T = P.mean(dim=0), Q.mean(dim=0)
        P_c, Q_c = P - centroid_P, Q - centroid_T 

        # Find rotation matrix by kabsch algorithm 
        H = P_c.T @ Q_c
        U, S, Vt = torch.linalg.svd(H)
        V = Vt.T

        d = torch.sign(torch.linalg.det(V @ U.T))

        diag_values = torch.cat(
            [
                torch.ones(1,dtype=P_c.dtype, device=P_c.device),
                torch.ones(1, dtype=P_c.dtype, device=P_c.device),
                d * torch.ones(1, dtype=P_c.dtype, device=P_c.device)
            ]
        )

        M = torch.eye(3, dtype=P_c.dtype, device=P_c.device) * diag_values
        R = V @ M @ U.T 
        
        # Find translation vector 
        t = centroid_T[None,:] - (R @ centroid_P[None,:].T).T
        t = t.T 
        return R, t.squeeze() 
    
    def kabsch_rmsd(
        self,
        pred_coords,
        true_coords,
        scale=True,
    ):
        pred_coords = pred_coords.squeeze(0)
        true_coords = true_coords.squeeze(0)
        if pred_coords.shape[0] != true_coords.shape[0]:
            raise ValueError("pred pos and target pos must have the same number of points")
        
        R, t = self.find_alignment(true_coords,pred_coords)
        true_coords_align = (R @ true_coords.T).T + t 
        true_coords_align = true_coords_align.detach()
        if scale:
            rmsd = torch.sqrt(((true_coords_align*10-pred_coords*10)**2).sum(axis=-1).mean())
        else:
            rmsd = torch.sqrt(((true_coords_align-pred_coords)**2).sum(axis=-1).mean())
        return rmsd, [rmsd]

    def rmsd_loss(
        self,
        pred_coords,
        true_coords,
        align=True,
        scale=True
    ):
        
        assert len(pred_coords.shape) == 3
        assert pred_coords.shape == true_coords.shape 

        if scale:
            pred_coords = pred_coords * 10 
            true_coords = true_coords * 10 

        total_rmsd = []
        for i in range(pred_coords.shape[0]):
            if align:
                rmsd = self.kabsch_rmsd(pred_coords[i],true_coords[i])
            else:
                rmsd = torch.sqrt(((pred_coords[i]-true_coords[i])**2).sum(axis=-1).mean())
            total_rmsd.append(rmsd)

        return (torch.stack(total_rmsd).sum()), total_rmsd


    def translation_loss(
        self,
        pred_coords,
        true_coords,
    ): 
        assert pred_coords.shape[0] == true_coords.shape[0]
        # pred_coords is shape T, N, 3 
        pred_displ = (pred_coords[1:,...] - pred_coords[:-1,...])*10
        true_displ = (true_coords[1:,...] - true_coords[:-1,...])*10

        trans_loss = F.l1_loss(pred_displ,true_displ)
        return trans_loss

    def centered_loss(
        self,
        pred_coords
        ):
            # pred_coords is shape [T,N,3]
            centered_loss = pred_coords.mean(axis=(0,1)).norm()
            return centered_loss

    def contact_loss(
        self,
        pred_coords,
        true_coords,
    ):
        distance_true = einx.subtract(
            "b i j, b n j -> b i n j", true_coords, true_coords
        ).norm(dim=-1) * 10 

        if pred_coords.shape[-1]==3:
            distance_pred = einx.subtract(
                "b i j, b n j -> b i n j", pred_coords, pred_coords
            ).norm(dim=-1) * 10
        else:
            distance_pred = pred_coords * 10 

        contact_loss = F.mse_loss(distance_pred, distance_true)
        return contact_loss

    def CA_distance_loss(self,pred_coords, true_coords):
        loss = 0
        for i in range(pred_coords.shape[0]):
            distance_pred = torch.cdist(pred_coords[i], pred_coords[i])
            distance_true = torch.cdist(true_coords[i], true_coords[i])

            ca_pred = torch.diagonal(distance_pred, offset=1)
            ca_true = torch.diagonal(distance_true, offset=1)
            diff = F.l1_loss(ca_pred, ca_true)
            loss = loss + diff

        return loss 
