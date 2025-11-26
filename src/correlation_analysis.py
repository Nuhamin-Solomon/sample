import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import utils

def analyze_correlation(ticker, sentiment_file, stock_data_folder):
    """
    Calculates and plots the correlation between daily news sentiment 
    and daily stock returns for a specific ticker.
    """
    # 1. Load Sentiment Data
    print(f"Loading sentiment data from {sentiment_file}...")
    df_sentiment = utils.load_csv_data(sentiment_file)
    
    # Ensure dates are datetime objects
    df_sentiment['trading_date'] = pd.to_datetime(df_sentiment['trading_date'])

    # Filter for the specific ticker (Column name is usually 'stock' in this dataset)
    if 'stock' not in df_sentiment.columns:
        print(f"Error: 'stock' column not found. Columns are: {df_sentiment.columns}")
        return

    df_ticker_sentiment = df_sentiment[df_sentiment['stock'] == ticker]
    
    if df_ticker_sentiment.empty:
        print(f"No news data found for ticker {ticker}.")
        return

    # Aggregate Sentiment: Calculate the MEAN sentiment score for each day
    daily_sentiment = df_ticker_sentiment.groupby('trading_date')['sentiment_score'].mean().reset_index()
    daily_sentiment.rename(columns={'sentiment_score': 'daily_sentiment'}, inplace=True)
    print(f"Found {len(daily_sentiment)} days of news for {ticker}.")

    # 2. Load Stock Data
    stock_path = f"{stock_data_folder}/{ticker}.csv"
    print(f"Loading stock data for {ticker}...")
    
    try:
        df_stock = utils.load_csv_data(stock_path)
    except FileNotFoundError:
        print(f"Stock file for {ticker} not found at {stock_path}.")
        return

    # Prepare Stock Data
    # Handle different case variations for 'Date'
    date_col = 'Date' if 'Date' in df_stock.columns else 'date'
    df_stock[date_col] = pd.to_datetime(df_stock[date_col])
    
    # Calculate Daily Returns (Percentage Change of Close Price)
    df_stock = df_stock.sort_values(date_col)
    df_stock['daily_return'] = df_stock['Close'].pct_change()

    # 3. Merge Datasets
    # Inner join: We only look at days where we have BOTH news and stock trading data
    merged_df = pd.merge(daily_sentiment, df_stock, left_on='trading_date', right_on=date_col, how='inner')
    
    print(f"Merged data has {len(merged_df)} data points.")

    if len(merged_df) < 2:
        print("Not enough overlapping data points to calculate correlation.")
        return

    # 4. Calculate Correlation
    correlation = merged_df['daily_sentiment'].corr(merged_df['daily_return'])
    print(f"\nPearson Correlation between Sentiment and Daily Return for {ticker}: {correlation:.4f}")

    # 5. Visualize
    plt.figure(figsize=(10, 6))
    sns.regplot(x='daily_sentiment', y='daily_return', data=merged_df, scatter_kws={'alpha':0.5})
    plt.title(f'Sentiment vs Stock Returns for {ticker} (Corr: {correlation:.2f})')
    plt.xlabel('Daily Sentiment Score')
    plt.ylabel('Daily Stock Return')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # CONFIGURATION
    SENTIMENT_FILE = 'data/clean/news_with_sentiment.csv' # The file you just created
    STOCK_FOLDER = 'data/raw/yfinance data' # Your stock data folder
    
    # Choose a ticker that exists in your news AND stock data
    # 'AAPL' is usually a safe bet if you have its CSV
    target_ticker = 'AAPL' 

    analyze_correlation(target_ticker, SENTIMENT_FILE, STOCK_FOLDER)