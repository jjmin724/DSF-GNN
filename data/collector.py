import pandas as pd
import yfinance as yf
from tqdm import tqdm
from core.utils import ensure_dir
from pathlib import Path
import json

class Collector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_dir = Path(cfg["data"]["raw_dir"])
        ensure_dir(self.raw_dir)
        with open("SnP500_list.json", "r", encoding="utf-8") as f:
            self.tickers = [c["symbol"] for c in json.load(f)["companies"]]

    def _download_one(self, ticker: str) -> pd.DataFrame:
        try:
            df = yf.download(ticker,
                             period=self.cfg["yfinance"]["period"],
                             interval=self.cfg["yfinance"]["interval"],
                             threads=self.cfg["yfinance"]["threads"],
                             progress=False)
            if df.empty:
                raise ValueError("empty dataframe")
            df["symbol"] = ticker
            return df
        except Exception as e:
            print(f"[Collector] {ticker} download error: {e}")
            return pd.DataFrame()

    def run(self):
        frames = []
        for tk in tqdm(self.tickers, desc="Collecting"):
            frames.append(self._download_one(tk))
        all_df = pd.concat(frames)
        out_path = self.raw_dir / "prices.parquet"
        all_df.to_parquet(out_path)
        print(f"[Collector] Saved raw prices to {out_path}")
