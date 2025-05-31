import joblib
import pandas as pd
import numpy as np

def test_model_prediction():
    # Load trained model
    model = joblib.load("models/best_xgb_model.pkl")

    sample_data=pd.read_csv(r'D:\Code files\Used Car Price Prediction\processed_data\preprocessed_data.csv').tail(1).drop(['price'],axis=1)
    # Make prediction
    prediction = model.predict(sample_data)

    # Assert output
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert prediction[0] > 0

