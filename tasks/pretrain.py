import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LassoLarsIC
import scipy.sparse as sp
from core.utils import ensure_dir
import json

class Pretrainer:
    """
    Granger-like 간략 전처리: 티커별 Lasso 기반 인과 Edge 추정 → dynamic_adj.npz 저장
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.feature_path = Path(cfg["data"]["processed_dir"]) / cfg["data"]["features_file"]
        self.graph_dir = Path(cfg["data"]["graph_dir"])
        ensure_dir(self.graph_dir)

    def _lagged_mat(self, series, lags):
        X = np.column_stack([series.shift(i) for i in range(1, lags+1)])
        return X

    def run(self):
        df = pd.read_parquet(self.feature_path)
        lags = self.cfg["model"]["lookback"]
        tickers = df["symbol"].unique().tolist()
        n = len(tickers)
        adj = sp.lil_matrix((n, n), dtype=np.float32)
        price_pivot = df.pivot(index="date", columns="symbol", values="close").fillna(method="ffill")
        for i, tgt in enumerate(tickers):
            y = price_pivot[tgt].pct_change().fillna(0).values
            X_list = []
            for src in tickers:
                X_list.append(self._lagged_mat(price_pivot[src].pct_change().fillna(0), lags))
            X_all = np.hstack(X_list)[lags:]
            y_all = y[lags:]
            try:
                model = LassoLarsIC(criterion="bic").fit(X_all, y_all)
                coefs = model.coef_.reshape(n, lags).sum(axis=1)
                parents = np.where(np.abs(coefs) > 1e-4)[0]
                for p in parents:
                    adj[p, i] = 1.0
            except Exception as e:
                print(f"[Pretrainer] Granger lite error on {tgt}: {e}")
        out_path = self.graph_dir / "dynamic_adj.npz"
        sp.save_npz(out_path, adj.tocsr())
        print(f"[Pretrainer] dynamic Granger graph saved -> {out_path}")
