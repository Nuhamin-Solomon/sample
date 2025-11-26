import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import pytz # Required for time zone conversion
from . import utils 

class SentimentAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
        # Download VADER lexicon quietly
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            
        self.analyzer = SentimentIntensityAnalyzer()

    def load_data(self):
        self.df = utils.load_csv_data(self.file_path)
        # Convert date to datetime with UTC timezone
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce', utc=True)
        return self.df

    def get_sentiment_score(self, text):
        if pd.isna(text):
            return 0.0
        return self.analyzer.polarity_scores(str(text))['compound']

    def _align_to_trading_day(self, dt):
        """
        Internal helper: Shifts a datetime to the correct trading date.
        - Converts to US/Eastern.
        - If time >= 4:00 PM (16:00), shift to next day.
        - Normalize to just the date component.
        """
        if pd.isna(dt):
            return None
            
        # 1. Convert to Eastern Time (Market Time)
        est = pytz.timezone('US/Eastern')
        dt_est = dt.astimezone(est)
        
        # 2. Define Market Close Hour (4 PM = 16:00)
        market_close_hour = 16
        
        # 3. Logic: If event is after market close, it affects TOMORROW
        trading_date = dt_est.date()
        if dt_est.hour >= market_close_hour:
            trading_date += timedelta(days=1)
            
        # 4. (Optional but good) If it falls on Saturday/Sunday, could shift to Monday
        # For now, we keep the calendar date; merging with stock data will handle gaps.
        
        return trading_date

    def analyze_sentiment(self, column_name='headline'):
        if self.df is None:
            self.load_data()
            
        print("1. Calculating sentiment scores...")
        self.df['sentiment_score'] = self.df[column_name].apply(self.get_sentiment_score)
        
        # Sentiment Labels
        self.df['sentiment_label'] = self.df['sentiment_score'].apply(
            lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
        )

        print("2. Aligning dates to trading days (Handling timezone & after-hours)...")
        # Apply the alignment logic
        self.df['trading_date'] = self.df['date'].apply(self._align_to_trading_day)
        
        # Convert trading_date back to datetime64 for easy merging later
        self.df['trading_date'] = pd.to_datetime(self.df['trading_date'])
        
        print("Analysis Complete. Sample Data:")
        print(self.df[['date', 'trading_date', 'headline', 'sentiment_score']].head())
        
        return self.df

    def save_results(self, output_path):
        """Saves the analyzed data to CSV."""
        if self.df is not None:
            # Uses the utils path helper if you want, or direct save
            full_path = utils.get_full_path(output_path)
            self.df.to_csv(full_path, index=False)
            print(f"Results saved to {full_path}")

if __name__ == "__main__":
    file_path = 'data/raw/news data/raw_analyst_ratings.csv'
    output_path = 'data/clean/news_with_sentiment.csv'
    
    analyzer = SentimentAnalyzer(file_path)
    df = analyzer.analyze_sentiment()
    analyzer.save_results(output_path)