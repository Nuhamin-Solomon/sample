import os
import sys
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import timedelta
import pytz
from typing import Optional

# Try relative import; if running as a script (no package context), add project root to sys.path and import utils
try:
    from . import utils
except Exception:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    import utils  # type: ignore

class SentimentAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None

        # Ensure VADER lexicon is available
        try:
            nltk.data.find("vader_lexicon")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        self.analyzer = SentimentIntensityAnalyzer()

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return os.path.abspath(path)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.normpath(os.path.join(project_root, path))

    def load_data(self) -> pd.DataFrame:
        resolved = self._resolve_path(self.file_path)
        if utils and hasattr(utils, "load_csv_data"):
            try:
                self.df = utils.load_csv_data(resolved)
            except Exception:
                self.df = pd.read_csv(resolved)
        else:
            self.df = pd.read_csv(resolved)

        # normalize date column
        date_cols = [c for c in self.df.columns if c.lower() in ("date", "datetime", "time")]
        if not date_cols:
            raise ValueError("No date/datetime column found in the input CSV.")
        date_col = date_cols[0]
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce", utc=True)
        self.df = self.df.rename(columns={date_col: "date"})
        return self.df

    def get_sentiment_score(self, text) -> float:
        if pd.isna(text):
            return 0.0
        try:
            return float(self.analyzer.polarity_scores(str(text))["compound"])
        except Exception:
            return 0.0

    def _align_to_trading_day(self, dt):
        if pd.isna(dt):
            return None
        ts = pd.to_datetime(dt, errors="coerce")
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize(pytz.UTC)
        est = pytz.timezone("US/Eastern")
        dt_est = ts.astimezone(est)
        trading_date = dt_est.date()
        if dt_est.hour >= 16:
            trading_date = trading_date + timedelta(days=1)
        return trading_date

    def analyze_sentiment(self, column_name: str = "headline") -> pd.DataFrame:
        if self.df is None:
            self.load_data()
        if column_name not in self.df.columns:
            candidates = [c for c in self.df.columns if c.lower() in ("headline", "title", "news", "text")]
            if not candidates:
                raise ValueError(f"Text column '{column_name}' not found.")
            column_name = candidates[0]

        self.df["sentiment_score"] = self.df[column_name].apply(self.get_sentiment_score)
        self.df["sentiment_label"] = self.df["sentiment_score"].apply(
            lambda s: "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
        )
        self.df["trading_date"] = self.df["date"].apply(self._align_to_trading_day)
        self.df["trading_date"] = pd.to_datetime(self.df["trading_date"])
        return self.df

    def save_results(self, output_path: str):
        if self.df is None:
            raise RuntimeError("No data to save. Run analyze_sentiment() first.")
        full = self._resolve_path(output_path) if not (utils and hasattr(utils, "get_full_path")) else utils.get_full_path(output_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        self.df.to_csv(full, index=False)
        print(f"Saved results to {full}")


if __name__ == "__main__":
    # Accept CLI path or search data/raw
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_raw = os.path.join(project_root, "data", "raw")

    file_path = None
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.isabs(file_path):
            file_path = os.path.normpath(os.path.join(project_root, file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Provided file not found: {file_path}")
    else:
        candidates = [
            os.path.join(data_raw, "news_headlines.csv"),
            os.path.join(data_raw, "News_Headlines.csv"),
            os.path.join(data_raw, "raw_analyst_ratings.csv"),
            os.path.join(data_raw, "news.csv"),
        ]
        file_path = next((p for p in candidates if os.path.exists(p)), None)
        if file_path is None and os.path.exists(data_raw):
            csvs = [f for f in os.listdir(data_raw) if f.lower().endswith(".csv")]
            if csvs:
                file_path = os.path.join(data_raw, csvs[0])

    if file_path is None:
        raise FileNotFoundError(f"No input CSV found under {data_raw}. Place your CSV there or pass path as argument.")

    output_path = os.path.join("data", "processed", "news_with_sentiment.csv")
    os.makedirs(os.path.dirname(os.path.join(project_root, output_path)), exist_ok=True)

    analyzer = SentimentAnalyzer(file_path)
    analyzer.analyze_sentiment()
    analyzer.save_results(output_path)