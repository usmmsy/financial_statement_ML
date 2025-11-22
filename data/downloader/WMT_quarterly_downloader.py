import yfinance as yf
import pandas as pd

# 1. Create a Ticker object for a specific stock
ticker_symbol = "COST"  # Replace with any stock ticker
end_data = "2025-01-01"
start_data = "2020-01-01"
ticker = yf.Ticker(ticker_symbol)

# 2. Retrieve the financial statements
# financials = ticker.financials
# balance_sheet = ticker.balance_sheet
# cash_flow = ticker.cash_flow
# You can also access quarterly data
quarterly_financials = ticker.quarterly_financials
quarterly_balance_sheet = ticker.quarterly_balance_sheet
quarterly_cash_flow = ticker.quarterly_cash_flow

# 3. Save to CSV files (optional)
# financials.to_csv(f"{ticker_symbol}_financials.csv")
# balance_sheet.to_csv(f"{ticker_symbol}_balance_sheet.csv")
# cash_flow.to_csv(f"{ticker_symbol}_cash_flow.csv")
quarterly_financials.to_csv(f"{ticker_symbol}_quarterly_financials.csv")
quarterly_balance_sheet.to_csv(f"{ticker_symbol}_quarterly_balance_sheet.csv")
quarterly_cash_flow.to_csv(f"{ticker_symbol}_quarterly_cash_flow.csv")

# 4. Print the dataframes (optional)
# print("--- Financials (Income Statement) ---")
# print(financials)
# print("\n--- Balance Sheet ---")
# print(balance_sheet)
# print("\n--- Cash Flow Statement ---")
# print(cash_flow)
print("\n--- Quarterly Financials ---")
print(quarterly_financials)
print("\n--- Quarterly Balance Sheet ---")
print(quarterly_balance_sheet)
print("\n--- Quarterly Cash Flow Statement ---")
print(quarterly_cash_flow)
