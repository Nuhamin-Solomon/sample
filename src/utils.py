import pandas as pd
import os

# Get the path to the project root (assuming utils.py is in src/)
# This path is relative: src/ -> project_root/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_full_path(relative_path):
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    return os.path.join(PROJECT_ROOT, relative_path)

def load_csv_data(relative_path, **kwargs):
    """
    Loads a CSV file into a Pandas DataFrame using the full path.
    
    Args:
        relative_path (str): Path relative to the project root.
        **kwargs: Optional arguments passed directly to pd.read_csv.
    """
    full_path = get_full_path(relative_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data file not found at: {full_path}")
    
    print(f"Loading data from: {relative_path}...")
    return pd.read_csv(full_path, **kwargs)