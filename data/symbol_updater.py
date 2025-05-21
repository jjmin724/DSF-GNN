import yfinance as yf
import json
import time
from pathlib import Path
from core.utils import ensure_dir

class SymbolUpdater:
    """
    최신 S&P500 편입 종목을 yfinance로 수집하여 SnP500_list.json(프로젝트 루트)에 저장.
    {
      "companies":[
        {"symbol":"AAPL","name":"Apple Inc."}, ...
      ]
    }
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.out_path = Path("SnP500_list.json")

    def _fetch_long_name(self, ticker: str) -> str:
        try:
            info = yf.Ticker(ticker).get_info()
            return info.get("longName") or info.get("shortName") or ticker
        except Exception:
            return ticker  # fallback

    def run(self):
        tickers = yf.tickers_sp500()
        companies = []
        for tk in tickers:
            name = self._fetch_long_name(tk)
            companies.append({"symbol": tk, "name": name})
            time.sleep(0.02)  # rate-limit 안전
        ensure_dir(self.out_path.parent)
        with self.out_path.open("w", encoding="utf-8") as f:
            json.dump({"companies": companies}, f, indent=2)
        print(f"[SymbolUpdater] {len(companies)}개 종목 저장 → {self.out_path}")
