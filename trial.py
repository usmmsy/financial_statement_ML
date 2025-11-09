import yfinance as yf

tickers = yf.Tickers('msft aapl goog')

tickers.tickers['MSFT'].info
tickers.tickers['AAPL'].info
tickers.tickers['GOOG'].info


