import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv 
from Modules.tensor_typing import Float, Int, Bool 

class TGCN(nn.Module):
    """
    https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/temporalgcn.py
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            improved=False,
            cached=False,
            add_self_loops=True):
        
        super(TGCN,self).__init__() 

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.improved = improved 
        self.cached = cached 
        self.add_self_loops = add_self_loops

        #Create update gate parameters and layers
        self.conv_z = GCNConv(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            improved = self.improved,
            cached = self.cached, 
            add_self_loops = self.add_self_loops)
        
        self.linear_z = nn.Linear(2 * out_channels, self.out_channels)

        #Create reset gate parameters and layers
        self.conv_r = GCNConv(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            improved = self.improved,
            cached = self.cached, 
            add_self_loops = self.add_self_loops)
        
        self.linear_r = nn.Linear(2 * out_channels, self.out_channels)

        #Create candidate state parameters and layers 
        self.conv_h = GCNConv(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            improved = self.improved,
            cached = self.cached, 
            add_self_loops = self.add_self_loops)
        
        self.linear_h = nn.Linear(2 * out_channels, self.out_channels)

    def _set_hidden_state(self,X,H):
        if H is None:
            H = torch.zeros(X.shape[0],self.out_channels).to(X.device)
        return H
    
    def _calculate_update_gate(self,X,edge_index,edge_weight,H):
        H = H.to(X.device)
        Z = torch.cat([self.conv_z(X,edge_index,edge_weight),H],dim=-1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z 
    
    def _calculate_reset_gate(self,X,edge_index,edge_weight,H):
        R = torch.cat([self.conv_r(X,edge_index,edge_weight),H],dim=-1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R
    
    def _calculate_candidate_state(self,X,edge_index,edge_weight,H,R):
        H_tilde = torch.cat([self.conv_h(X,edge_index,edge_weight), H * R],dim=-1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde
    
    def _calculate_hidden_state(self,Z,H,H_tilde):
        H = Z * H + (1-Z) * H_tilde
        return H
    
    def forward(
            self,
            X:Float['T N D'],
            edge_index:torch.LongTensor,
            edge_weight:torch.FloatTensor = None,
            H: torch.FloatTensor = None): 
        
        H = self._set_hidden_state(X,H)
        Z = self._calculate_update_gate(X,edge_index,edge_weight,H)
        R = self._calculate_reset_gate(X,edge_index,edge_weight,H)
        H_tilde = self._calculate_candidate_state(X,edge_index,edge_weight,H,R)
        H = self._calculate_hidden_state(Z,H,H_tilde)
        return H 
    

class AttTGCN(nn.Module):
    """
    
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            periods:int = 10,
            improved=False,
            cached = False,
            add_self_loops = True,
            num_nodes=220): 
        
        super(AttTGCN,self).__init__() 
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self.base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learned_attention = nn.Linear(in_channels,1)

    def forward(
            self,
            X:Float['T N D'],
            edge_index:torch.LongTensor,
            edge_weight:torch.FloatTensor=None,
            H:torch.FloatTensor=None): 
        
        assert X.shape[0] == self.periods
        H_accum = 0 
        X = X.transpose(0,1)  # Shape [N, T, D ]
        probs = self.learned_attention(X).squeeze(-1)
        X = X.transpose(0,1) # Back to shape [T,N,D]
        probs = F.softmax(probs,dim=1).unsqueeze(-1) ## Provide node importance across time 
        for period in range(self.periods):
            if edge_weight is not None:
                edge_attr = edge_weight[period]
            else:
                edge_attr=None
            H = self.base_tgcn(
                X[period,:,:],edge_index,edge_attr,H
            )
            H_accum = H_accum + probs[:,period] * H
        
        return H_accum 