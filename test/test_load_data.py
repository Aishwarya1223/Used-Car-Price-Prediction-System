from pathlib import Path
import pandas as pd
from train import load_data

def test_load_data_success():
    df = load_data(Path("new_data/new_data.csv"))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
