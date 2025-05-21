import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from models.dsf_gnn import DSFGNN
from core.utils import set_seed
import json

class PriceDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        data = pd.read_parquet(Path(cfg["data"]["processed_dir"]) /
                               cfg["data"]["features_file"])
        self.lookback = cfg["model"]["lookback"]
        self.tickers = data["symbol"].unique().tolist()
        pivot = data.pivot(index="date", columns="symbol",
                           values=["open", "high", "low", "close", "volume"]).dropna()
        self.X = []
        self.y = []
        dates = pivot.index.tolist()
        mat = pivot.values.reshape(len(dates), len(self.tickers), -1)
        for i in range(self.lookback, len(dates)-1):
            self.X.append(mat[i-self.lookback:i])
            up = (pivot["close"].iloc[i+1] >= pivot["close"].iloc[i]).astype(float).values
            self.y.append(up)
        split = int(len(self.X)*0.8)
        if mode == "train":
            self.X = self.X[:split]
            self.y = self.y[:split]
        else:
            self.X = self.X[split:]
            self.y = self.y[split:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)

class Trainer:
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        set_seed(cfg["random_seed"])
        self.dataset = PriceDataset(cfg, mode="train" if mode == "train" else "test")
        self.loader = DataLoader(self.dataset, batch_size=cfg["model"]["batch_size"],
                                 shuffle=(mode=="train"))
        feat_dim = 5  # open high low close volume
        self.model = DSFGNN(cfg,
                            num_nodes=len(self.dataset.tickers),
                            feat_dim=feat_dim).cuda()
        self.opt = optim.Adam(self.model.parameters(), lr=cfg["model"]["lr"])
        self.crit = nn.BCELoss()

        # static graphs
        gdir = Path(cfg["data"]["graph_dir"])
        self.static_edges = []
        for name in ["industry", "shareholding"]:
            adj = sp.load_npz(gdir / f"{name}.npz").tocoo()
            edge_index = torch.tensor(np.vstack((adj.row, adj.col)),
                                      dtype=torch.long).cuda()
            self.static_edges.append(edge_index)

    def run(self):
        if self.mode == "train":
            self._train_loop()
        else:
            self._predict()

    def _train_loop(self):
        for epoch in range(self.cfg["model"]["epochs"]):
            self.model.train()
            total_loss = 0
            for X, y in self.loader:
                batch = {
                    "x": X.cuda(),              # (B, N, T, D)
                    "static_edges": self.static_edges
                }
                prob = self.model(batch)        # (B, N)
                loss = self.crit(prob, y.cuda())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
            print(f"[Trainer] epoch {epoch+1}/{self.cfg['model']['epochs']} "
                  f"loss={total_loss/len(self.loader):.4f}")
        torch.save(self.model.state_dict(),
                   Path(self.cfg["workspace"]) / "model.pt")

    def _predict(self):
        ckpt = Path(self.cfg["workspace"]) / "model.pt"
        if not ckpt.exists():
            print("[Predict] model checkpoint not found.")
            return
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X, _ in self.loader:
                batch = {"x": X.cuda(), "static_edges": self.static_edges}
                prob = self.model(batch).cpu().numpy()
                preds.append(prob)
        preds = np.vstack(preds)
        out_path = Path(self.cfg["workspace"]) / "predictions.npy"
        np.save(out_path, preds)
        print(f"[Predict] Saved probability predictions → {out_path}")
