import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from core.utils import ensure_dir
from collections import defaultdict
import yfinance as yf
import json
import scipy.sparse as sp

class GraphBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc_dir = Path(cfg["data"]["processed_dir"])
        self.graph_dir = Path(cfg["data"]["graph_dir"])
        ensure_dir(self.graph_dir)
        with open("SnP500_list.json", "r", encoding="utf-8") as f:
            self.company_info = json.load(f)["companies"]

    # ----- Static graph: industry -----
    def _industry_edges(self):
        edges = []
        industry_map = defaultdict(list)
        for comp in self.company_info:
            try:
                info = yf.Ticker(comp["symbol"]).get_info()
                industry = info.get("industry") or "Unknown"
                industry_map[industry].append(comp["symbol"])
            except Exception:
                continue
        for _, symbols in industry_map.items():
            for i, a in enumerate(symbols):
                for b in symbols[i+1:]:
                    edges.append((a, b, 1.0))
        return edges

    # example shareholding overlap via common top institution (약식)
    def _share_edges(self):
        edges = []
        # 간단화를 위해 동일 섹터 공통  ETF 로 가정 (dummy weight 0.5)
        sector_map = defaultdict(list)
        for comp in self.company_info:
            try:
                info = yf.Ticker(comp["symbol"]).get_info()
                sec = info.get("sector") or "Unknown"
                sector_map[sec].append(comp["symbol"])
            except Exception:
                continue
        for _, symbols in sector_map.items():
            for i, a in enumerate(symbols):
                for b in symbols[i+1:]:
                    edges.append((a, b, 0.5))
        return edges

    def _save_adj(self, edges, name, nodes):
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)
        adj = nx.to_scipy_sparse_array(G, dtype=np.float32, nodelist=nodes)
        path = self.graph_dir / f"{name}.npz"
        sp.save_npz(path, adj)
        print(f"[GraphBuilder] {name} graph saved ({adj.nnz} edges) -> {path}")

    def run(self):
        nodes = [c["symbol"] for c in self.company_info]
        self._save_adj(self._industry_edges(), "industry", nodes)
        self._save_adj(self._share_edges(), "shareholding", nodes)
        # media graph 생략(뉴스 크롤링 부하). 필요시 추가 구현.
