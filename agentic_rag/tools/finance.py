from typing import Optional, List
import yfinance as yf

FinanceResult = dict   # {"symbol": str, "company_name": str, "current_price": float, "historical_prices": list, "market_cap": int}


def fetch_stock_data(symbol: str, period: str = "3mo") -> Optional[FinanceResult]:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period=period)
        if hist.empty:
            return None

        prices = [
            {"date": str(idx.date()), "close": round(row["Close"], 2)}
            for idx, row in hist.iterrows()
        ]
        return {
            "symbol": symbol.upper(),
            "company_name": info.get("shortName", symbol),
            "current_price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "historical_prices": prices[-90:],   # last 90 trading days max
        }
    except Exception:
        return None
