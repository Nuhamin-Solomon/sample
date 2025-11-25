import pandas as pd
import talib
import matplotlib.pyplot as plt
import os

class StockAnalyzer:
    def __init__(self, data_folder):
        """
        Initialize with the folder where stock CSVs are located.
        """
        self.data_folder = data_folder
        self.df = None

    def load_stock_data(self, ticker):
        """
        Load a specific stock CSV (e.g., AAPL.csv) from the data folder.
        """
        # Construct the file path (assuming file is named like 'AAPL.csv')
        file_path = os.path.join(self.data_folder, f"{ticker}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Stock file not found: {file_path}")
        
        print(f"Loading data for {ticker}...")
        self.df = pd.read_csv(file_path)
        
        # Ensure date is datetime and set as index
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
        elif 'date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['date'])
            self.df.set_index('Date', inplace=True)
            
        print(f"Loaded {len(self.df)} rows for {ticker}.")
        return self.df

    def calculate_technical_indicators(self):
        """
        Calculate SMA, RSI, and MACD using TA-Lib.
        """
        if self.df is None:
            print("Data not loaded.")
            return

        print("Calculating Technical Indicators...")
        
        # 1. Simple Moving Average (SMA) - 20 days
        self.df['SMA_20'] = talib.SMA(self.df['Close'], timeperiod=20)
        
        # 2. Relative Strength Index (RSI) - 14 days
        self.df['RSI'] = talib.RSI(self.df['Close'], timeperiod=14)
        
        # 3. MACD (Moving Average Convergence Divergence)
        # macd = MACD line, macdsignal = Signal line, macdhist = MACD Histogram
        self.df['MACD'], self.df['MACD_signal'], self.df['MACD_hist'] = talib.MACD(
            self.df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        print("Indicators calculated successfully.")

    def plot_analysis(self, ticker):
        """
        Plot the Price, SMA, RSI, and MACD.
        """
        if self.df is None: 
            return

        plt.figure(figsize=(14, 10))

        # Subplot 1: Price and SMA
        plt.subplot(3, 1, 1)
        plt.plot(self.df.index, self.df['Close'], label='Close Price', color='blue')
        plt.plot(self.df.index, self.df['SMA_20'], label='SMA (20)', color='orange', linestyle='--')
        plt.title(f'{ticker} Stock Price with SMA')
        plt.legend()
        plt.grid(True)

        # Subplot 2: RSI
        plt.subplot(3, 1, 2)
        plt.plot(self.df.index, self.df['RSI'], label='RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--', alpha=0.5) # Overbought
        plt.axhline(30, color='green', linestyle='--', alpha=0.5) # Oversold
        plt.title('Relative Strength Index (RSI)')
        plt.legend()
        plt.grid(True)

        # Subplot 3: MACD
        plt.subplot(3, 1, 3)
        plt.plot(self.df.index, self.df['MACD'], label='MACD', color='blue')
        plt.plot(self.df.index, self.df['MACD_signal'], label='Signal Line', color='red')
        plt.bar(self.df.index, self.df['MACD_hist'], label='MACD Histogram', color='gray', alpha=0.3)
        plt.title('MACD')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
   
    DATA_FOLDER = 'data/raw/prices' 
    
    # CHANGE THIS to a stock symbol that exists in your folder (e.g., 'AAPL', 'TSLA')
    TICKER_SYMBOL = 'AAPL'  
    
    analyzer = StockAnalyzer(DATA_FOLDER)
    
    try:
        analyzer.load_stock_data(TICKER_SYMBOL)
        analyzer.calculate_technical_indicators()
        analyzer.plot_analysis(TICKER_SYMBOL)
    except Exception as e:
        print(f"Error: {e}")

        