import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os

def get_institutional_investors_data():
    """
    Placeholder function to integrate Three Primary Institutional Investors (三大法人) data.
    This includes:
      - Foreign Investors (外資)
      - Investment Trusts (投信)
      - Dealers (自營商)
    """
    # TODO: Implement API request or logic to scrape institutional investors' data from TWSE.
    # Example: fetch from Taiwan Stock Exchange OpenAPI and append to dataframe
    pass

def calculate_rsi(series, window=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Use Wilder's smoothing method (exponential moving average) for RSI
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def collect_and_process_data(ticker='0050.TW', start_date='2021-02-24', end_date=None):
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        print("No data retrieved. Please check the ticker symbol or date ranges.")
        return None
        
    # In newer versions of yfinance, the returned dataframe may have a MultiIndex column.
    # Flatten it to just the standard ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    print("Downloading TAIEX (^TWII) reference data...")
    taiex_df = yf.download('^TWII', start=start_date, end=end_date)
    if not taiex_df.empty:
        if isinstance(taiex_df.columns, pd.MultiIndex):
            taiex_df.columns = taiex_df.columns.get_level_values(0)
        df['TAIEX_Close'] = taiex_df['Close']
    else:
        df['TAIEX_Close'] = np.nan
        
    print("Calculating technical indicators...")
    close_price = df['Close']
    
    # Taiwan Market commonly uses MA5, MA10, MA20 (Monthly Line) and MA60 (Quarterly Line)
    df['MA5'] = close_price.rolling(window=5).mean()
    df['MA10'] = close_price.rolling(window=10).mean()
    df['MA20'] = close_price.rolling(window=20).mean()
    df['MA60'] = close_price.rolling(window=60).mean()
    
    # Calculate RSI(14)
    df['RSI14'] = calculate_rsi(close_price, window=14)
    
    # Call the placeholder for institutional data integration
    get_institutional_investors_data()
    
    print("Cleaning data...")
    # Data Cleaning:
    # 1. Forward-fill any accidental missing values during trading days.
    df = df.ffill()
    # 2. Drop rows with null values (especially the first 60 days which lack the MA60 values).
    # This prepares clean sliding-window datasets for LSTM.
    df = df.dropna()
    
    # Re-verify no NaNs are passed forwards
    assert df.isnull().sum().sum() == 0, "Data still contains Null values after cleaning!"
    
    # Save the processed data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Data will preserve the 'Date' index which acts as part of our timestamp requirement
    output_filepath = os.path.join(output_dir, 'taiwan_stock_processed.csv')
    df.to_csv(output_filepath)
    print(f"Data saved to {output_filepath}")
    
    return df

if __name__ == "__main__":
    # Test execution for Yuanta Taiwan 50 ETF (0050.TW)
    stock_data = collect_and_process_data('0050.TW')
    if stock_data is not None:
        print("Dataset Head:")
        print(stock_data.head())
