import pandas as pd
from train import preprocessing
import joblib

def test_preprocessing_success():
    df = pd.read_csv("new_data/new_data.csv")
    processed_df = preprocessing(df)
    
    assert isinstance(processed_df, pd.DataFrame)
    assert "model_encoded" in processed_df.columns
    assert "price" in processed_df.columns
