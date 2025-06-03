import pandas as pd
from xgboost import XGBRegressor
import joblib
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import os
from pathlib import Path
from datetime import datetime
import h2o
from h2o.automl import H2OAutoML
import json
import shutil
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pickle

# Load environment variables
load_dotenv()
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
host = os.getenv("MYSQL_HOST")
port = os.getenv("MYSQL_PORT")
db = os.getenv("MYSQL_DB")

# Create SQLAlchemy engine
engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db}")

# Query recent records from car_data table
query = "SELECT * FROM car_data WHERE updated_at >= NOW() - INTERVAL 7 DAY"
df = pd.read_sql(query, engine)

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    if 'updated_at' in df.columns:
        df = df.drop(columns=['updated_at'])

    ohe_cols = ['transmission', 'fuelType']
    with open('picklefile_preprocessors/onehot_encoder.pkl', 'rb') as f:
        ohe = pickle.load(f)
    ohe_encoded = ohe.transform(df[ohe_cols])
    ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)

    df_encoded = df.drop(columns=ohe_cols).reset_index(drop=True)
    df_encoded = pd.concat([df_encoded, ohe_encoded_df.reset_index(drop=True)], axis=1)

    with open('picklefile_preprocessors/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    target_encoded_df = target_encoder.transform(df[['model', 'brand']].copy())
    df_encoded = pd.concat([df_encoded, target_encoded_df.reset_index(drop=True)], axis=1)
    df_encoded.drop(columns=['model', 'brand'], inplace=True)

    return df_encoded

if __name__ == "__main__":
    nrows = df.shape[0]
    if nrows >= 20:
        nfolds = 5
        max_models = 20
    elif nrows >= 2:
        nfolds = -1
        max_models = 5
    else:
        print("Not enough data to train")
        exit()

    df_processed = preprocessing(df)

    x = df_processed.drop('price', axis=1)
    y = df_processed['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Save feature order used during training
    
    with open("picklefile_preprocessors/feature_names.pkl", "wb") as f:
        pickle.dump(list(x_train.columns), f)

    model = joblib.load('models/best_xgb_model.pkl')

    train_df = pd.concat([x_train, y_train.rename("price")], axis=1)
    test_df = pd.concat([x_test, y_test.rename("price")], axis=1)

    h2o.init(start_h2o=True, nthreads=-1, max_mem_size="2G", port=54321)
    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    features = x.columns.tolist()
    target = "price"

    aml = H2OAutoML(max_models=max_models, seed=42, nfolds=nfolds)
    aml.train(x=features, y=target, training_frame=train_h2o)

    perf = aml.leader.model_performance(test_data=test_h2o)
    r2_h2o = perf.r2()
    rmse_h2o = perf.rmse()
    h2o_model_path = h2o.save_model(model=aml.leader, path="models", force=True)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_model_path = f"models/xgb_model_{timestamp}.pkl"
    joblib.dump(model, local_model_path)

    if r2 > r2_h2o:
        print("XGBoost is the best model.")
        best_model_type = "XGBoost"
        best_model_path = local_model_path
    elif r2 < r2_h2o:
        print("H2O AutoML is the best model.")
        best_model_type = "H2OAutoML"
        best_model_path = h2o_model_path
    else:
        print("Both models have equal R². Choosing XGBoost by default.")
        best_model_type = "XGBoost"
        best_model_path = local_model_path

    if os.path.exists("best_model"):
        shutil.rmtree("best_model")
    os.makedirs("best_model", exist_ok=True)
    shutil.copy(best_model_path, "best_model/best_model.pkl")

    metadata = {
        "best_model": best_model_type,
        "r2_score": max(r2, r2_h2o),
        "timestamp": timestamp
    }
    with open("best_model/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    h2o.shutdown(prompt=False)
