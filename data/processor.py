import pandas as pd
from pathlib import Path
from core.utils import ensure_dir

class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_path = Path(cfg["data"]["raw_dir"]) / "prices.parquet"
        self.proc_dir = Path(cfg["data"]["processed_dir"])
        ensure_dir(self.proc_dir)

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 필요한 컬럼만 rename
        df = df.reset_index().rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        })
        # 추가 파생예시: 일간 수익률
        df["return"] = df.groupby("symbol")["close"].pct_change().fillna(0)
        return df

    def run(self):
        df = pd.read_parquet(self.raw_path)
        feats = self._compute_features(df)
        out_file = self.proc_dir / self.cfg["data"]["features_file"]
        feats.to_parquet(out_file)
        print(f"[Preprocessor] Features saved to {out_file}")
