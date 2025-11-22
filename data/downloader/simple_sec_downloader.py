"""
SIMPLE SEC EDGAR DOWNLOADER
============================

This script downloads 10+ years of financial data from SEC EDGAR in 3 steps:
1. Change your email address below
2. Change the ticker symbol
3. Run the script
"""

import requests
import pandas as pd
import json
import time

# ============================================
# CONFIGURATION - CHANGE THESE
# ============================================

YOUR_EMAIL = "usmmsy@gmail.com"  # SEC requires this
TICKER = "WMT"  # Change to any ticker you want

# ============================================

def download_sec_data(ticker, email):
    """
    Download financial statements from SEC EDGAR
    Returns 10+ years of annual data
    """
    
    print(f"\n{'='*70}")
    print(f"Downloading SEC EDGAR data for {ticker}")
    print('='*70)
    
    # Step 1: Get CIK (company identifier)
    print("\nStep 1: Getting company CIK...")
    headers = {
        'User-Agent': f'{email}',
        'Accept-Encoding': 'gzip, deflate',
    }
    
    # Get ticker to CIK mapping
    mapping_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(mapping_url, headers=headers)
    companies = response.json()
    
    cik = None
    for key, company in companies.items():
        if company['ticker'].upper() == ticker.upper():
            cik = str(company['cik_str']).zfill(10)
            company_name = company['title']
            print(f"✓ Found: {company_name}")
            print(f"  CIK: {cik}")
            break
    
    if not cik:
        print(f"✗ Ticker {ticker} not found")
        return None
    
    # Step 2: Download company facts (all financial data)
    print("\nStep 2: Downloading financial data from SEC...")
    time.sleep(0.1)  # Be polite to SEC servers
    
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(facts_url, headers=headers)
    
    if response.status_code != 200:
        print(f"✗ Error: {response.status_code}")
        return None
    
    company_facts = response.json()
    print(f"✓ Downloaded data for {company_facts['entityName']}")
    
    # Step 3: Extract and structure the data
    print("\nStep 3: Extracting financial statements...")
    
    us_gaap = company_facts['facts']['us-gaap']
    
    # Key financial items to extract
    key_items = {
        # Income Statement
        'Revenues': 'Total Revenue',
        'RevenueFromContractWithCustomerExcludingAssessedTax': 'Revenue',
        'CostOfRevenue': 'Cost of Revenue',
        'GrossProfit': 'Gross Profit',
        'OperatingIncomeLoss': 'Operating Income',
        'NetIncomeLoss': 'Net Income',
        'EarningsPerShareBasic': 'EPS Basic',
        'EarningsPerShareDiluted': 'EPS Diluted',
        
        # Balance Sheet
        'Assets': 'Total Assets',
        'AssetsCurrent': 'Current Assets',
        'CashAndCashEquivalentsAtCarryingValue': 'Cash',
        'Liabilities': 'Total Liabilities',
        'LiabilitiesCurrent': 'Current Liabilities',
        'StockholdersEquity': 'Stockholders Equity',
        
        # Cash Flow
        'NetCashProvidedByUsedInOperatingActivities': 'Operating Cash Flow',
        'NetCashProvidedByUsedInInvestingActivities': 'Investing Cash Flow',
        'NetCashProvidedByUsedInFinancingActivities': 'Financing Cash Flow',
        'PaymentsToAcquirePropertyPlantAndEquipment': 'CapEx',
    }
    
    # Collect all data
    all_data = []
    
    for gaap_name, friendly_name in key_items.items():
        if gaap_name in us_gaap:
            item = us_gaap[gaap_name]
            
            # Get USD values
            if 'units' in item and 'USD' in item['units']:
                for entry in item['units']['USD']:
                    # Only get annual reports (10-K)
                    if entry.get('form') == '10-K':
                        all_data.append({
                            'Date': entry['end'],
                            'Item': friendly_name,
                            'Value': entry['val'],
                            'Fiscal Year': entry.get('fy', ''),
                        })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        print("✗ No data found")
        return None
    
    # Pivot: dates as rows, items as columns
    df['Date'] = pd.to_datetime(df['Date'])
    pivot_df = df.pivot_table(
        index='Date',
        columns='Item',
        values='Value',
        aggfunc='first'
    )
    pivot_df = pivot_df.sort_index()
    
    print(f"✓ Extracted financial data:")
    print(f"  Years: {len(pivot_df)}")
    print(f"  Date range: {pivot_df.index[0].strftime('%Y-%m-%d')} to {pivot_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Line items: {len(pivot_df.columns)}")
    
    # Save to CSV
    filename = f"{ticker}_financials_SEC_EDGAR.csv"
    pivot_df.to_csv(filename)
    print(f"\n✓ Saved to: {filename}")
    
    # Also save raw JSON
    json_filename = f"{ticker}_raw_data_SEC.json"
    with open(json_filename, 'w') as f:
        json.dump(company_facts, f, indent=2)
    print(f"✓ Raw data saved to: {json_filename}")
    
    # Display preview
    print(f"\n{'='*70}")
    print("DATA PREVIEW:")
    print('='*70)
    print(pivot_df.to_string())
    print('='*70)
    
    return pivot_df


# ============================================
# RUN THE SCRIPT
# ============================================

if __name__ == "__main__":
    
    if YOUR_EMAIL == "your.email@example.com":
        print("\n⚠️  WARNING: Please change YOUR_EMAIL in the script!")
        print("   The SEC requires a valid email address.\n")
    
    # Download data
    df = download_sec_data(TICKER, YOUR_EMAIL)
    
    if df is not None:
        print("\n✅ SUCCESS! Your data has been downloaded.")
        print(f"\nYou now have {len(df)} years of financial data for {TICKER}")
        print("\nFiles created:")
        print(f"  1. {TICKER}_financials_SEC_EDGAR.csv  (structured data)")
        print(f"  2. {TICKER}_raw_data_SEC.json  (raw API response)")
        
        # Show what columns are available
        print(f"\nAvailable metrics:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
    else:
        print("\n✗ Download failed. Please check the ticker symbol and try again.")
