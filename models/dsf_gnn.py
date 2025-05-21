import torch
import torch.nn as nn
from models.modules import TFR, AttentionGCN, RFF
from torch_geometric.utils import add_self_loops

class DSFGNN(nn.Module):
    def __init__(self, cfg, num_nodes, feat_dim):
        super().__init__()
        h = cfg["model"]["hidden_dim"]
        self.lookback = cfg["model"]["lookback"]
        self.tfr = TFR(feat_dim, h, num_layers=1, dropout=cfg["model"]["dropout"])
        # two static relations (industry, shareholding)
        self.sre1 = AttentionGCN(h, h)
        self.sre2 = AttentionGCN(h, h)
        self.rff = RFF(h, h, h)
        self.pred = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(cfg["model"]["dropout"]),
            nn.Linear(h, 1)  # sigmoid 확률
        )

    def forward(self, batch):
        """
        batch dict keys:
          x   (B*N, T, feat)
          static_edges : list of edge_index tensors len=2
          edge_weights : list of weights
        """
        B, N, T, D = batch["x"].shape
        x = batch["x"].view(B*N, T, D)
        seq, final = self.tfr(x)                            # (B*N, T, h), (B*N,h)
        # Static GCN embeddings
        edge1 = add_self_loops(batch["static_edges"][0])[0]
        s1 = self.sre1(final, edge1)
        edge2 = add_self_loops(batch["static_edges"][1])[0]
        s2 = self.sre2(final, edge2)
        s = (s1 + s2) / 2.0
        d = final                                           # 간단히 TFR 최종 상태를 dynamic emb 로 사용
        f = self.rff(d, s)
        logits = self.pred(f).view(B, N)
        prob_up = torch.sigmoid(logits)
        return prob_up
