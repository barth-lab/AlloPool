import torch
import torch.nn as nn 
import torch.nn.functional as F 


class SelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        key_dim,
        mlp_dim,
        n_head,
    ):
        super(SelfAttention, self).__init__()

        self.attention = nn.MultiheadAttention(
            input_dim, n_head, dropout=0.1, kdim=key_dim, batch_first=True
        )

        self.input_dim=input_dim
        self.key_dim=key_dim
        self.n_head = n_head
        self.mlp_dim = mlp_dim

        self.Wq = nn.Linear(self.input_dim, self.input_dim)
        self.Wk = nn.Linear(self.input_dim, self.key_dim)
        self.Wv = nn.Linear(self.input_dim, self.input_dim)

        self.norm1 = nn.LayerNorm(self.input_dim)
        self.norm2 = nn.LayerNorm(self.input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.input_dim)
        )

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        y, _ = self.attention(k, q, v)

        y = self.norm1(x + y)
        y = self.norm2(y + self.mlp(y))

        return y 

