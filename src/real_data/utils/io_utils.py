import json
import os
import pandas as pd
from typing import Any, Dict, Tuple

def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file with standard formatting."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def save_csv(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8")

def load_csv(file_path: str) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    return pd.read_csv(file_path)

def create_directory(path: str, clean: bool = False) -> None:
    """Create directory, optionally cleaning if it exists."""
    if clean and os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
