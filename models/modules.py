import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, MessagePassing, global_add_pool

class TFR(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout, bidirectional=False)
    def forward(self, x):
        # x: (B, T, D)
        out, h = self.gru(x)
        return out, h[-1]  # sequence, final state

class AttentionGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim*2, 1)
        self.conv = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        # simple attention weight on edges
        row, col = edge_index
        attn = torch.sigmoid(
            self.lin(torch.cat([x[row], x[col]], dim=1))).squeeze()
        if edge_weight is not None:
            attn = attn * edge_weight
        out = self.conv(x, edge_index, attn)
        return out

class RFF(nn.Module):
    def __init__(self, d_dyn, d_stat, out_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(d_dyn, d_stat, out_dim)
        self.lin = nn.Linear(d_dyn + d_stat, out_dim)

    def forward(self, d, s):
        return torch.tanh(self.bilinear(d, s) + self.lin(torch.cat([d, s], dim=-1)))
