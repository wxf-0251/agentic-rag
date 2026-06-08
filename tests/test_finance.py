from unittest.mock import patch, MagicMock
import pandas as pd
from agentic_rag.tools.finance import fetch_stock_data, FinanceResult


def test_fetch_stock_data_returns_structured():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({
        "Close": [150.0, 152.0, 149.0],
    }, index=pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"]))
    mock_ticker.info = {"shortName": "Tesla Inc", "currentPrice": 152.0, "marketCap": 500000000000}

    with patch("agentic_rag.tools.finance.yf.Ticker") as mock_ticker_class:
        mock_ticker_class.return_value = mock_ticker
        result = fetch_stock_data("TSLA", period="3mo")

    assert result is not None
    assert result["symbol"] == "TSLA"
    assert result["company_name"] == "Tesla Inc"
    assert result["current_price"] == 152.0
    assert len(result["historical_prices"]) == 3
    assert result["historical_prices"][0]["date"] == "2026-01-02"


def test_fetch_stock_data_handles_invalid_symbol():
    with patch("agentic_rag.tools.finance.yf.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker.info = {"trailingPegRatio": None}
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        result = fetch_stock_data("INVALID123XX")
        assert result is None


def test_fetch_stock_data_handles_exception():
    with patch("agentic_rag.tools.finance.yf.Ticker") as mock_ticker_class:
        mock_ticker_class.side_effect = Exception("Network error")
        result = fetch_stock_data("TSLA")
        assert result is None
