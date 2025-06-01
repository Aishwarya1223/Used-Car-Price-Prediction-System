import pandas as pd
from xgboost import XGBRegressor
import joblib
import category_encoders as ce
from  sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error
import h2o
from h2o.automl import H2OAutoML
import json,shutil
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

def load_data(path:Path) -> pd.DataFrame:
    if os.path.exists(path):
        df=pd.read_csv(path)
        return df
    else:
        raise FileNotFoundError(f'No file at path: {path}')


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    ohe_cols = ['transmission', 'fuelType']

    # Load pre-trained OneHotEncoder
    with open('picklefile_preprocessors/onehot_encoder.pkl', 'rb') as f:
        ohe = pickle.load(f)

    # Transform categorical features
    ohe_encoded = ohe.transform(df[ohe_cols])
    ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)

    # Drop original categorical columns and concatenate encoded ones
    df_encoded = df.drop(columns=ohe_cols).reset_index(drop=True)
    df_encoded = pd.concat([df_encoded, ohe_encoded_df.reset_index(drop=True)], axis=1)

    # Load pre-trained TargetEncoder
    with open('picklefile_preprocessors/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)

    # Transform 'model' and 'brand' using the target encoder
    target_encoded_df = target_encoder.transform(df[['model', 'brand']].copy())
    
    # Concatenate target encoded columns
    df_encoded = pd.concat([df_encoded, target_encoded_df.reset_index(drop=True)], axis=1)

    # Drop original 'model' and 'brand' columns
    df_encoded.drop(columns=['model', 'brand'], inplace=True)

    return df_encoded



if __name__=="__main__":
    
    #df=load_data(Path('new_data/new_data.csv'))
    df_processed=preprocessing(df)
    
    x=df_processed.drop('price',axis=1)
    y=df_processed['price']
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    # Xgboost
    model: XGBRegressor = joblib.load('models/best_xgb_model.pkl')
    
    # h2o model
    train_df = pd.concat([x_train, y_train.rename("price")], axis=1)
    test_df = pd.concat([x_test, y_test.rename("price")], axis=1)

    h2o.init(start_h2o=True, nthreads=-1, max_mem_size="2G", port=54321)
    
    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    features = x.columns.tolist()
    target = "price"

    # Train H2O AutoML
    aml = H2OAutoML(max_models=10, seed=42)
    aml.train(x=features, y=target, training_frame=train_h2o)

    # Evaluate
    perf = aml.leader.model_performance(test_data=test_h2o)
    r2_h2o = perf.r2()
    rmse_h2o = perf.rmse()

    # Save model
    h2o_model_path = h2o.save_model(model=aml.leader, path="models", force=True)
    
    
    
    with mlflow.start_run(run_name="TrainingPipeline") as parent_run:
        with mlflow.start_run(run_name="XGBRegressor", nested=True):
            # Train
            model.fit(x_train, y_train)

            # Evaluate
            y_pred = model.predict(x_test)
            rmse = root_mean_squared_error(y_test, y_pred) 
            r2 = r2_score(y_test, y_pred)

            # Log params and metrics
            mlflow.log_param("model_type", "XGBRegressor")
            mlflow.log_param("train_size", len(x_train))
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
            

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path="xgb_model")

            # Log encoders
            mlflow.log_artifact("picklefile_preprocessors/ohe_encoder.pkl")
            mlflow.log_artifact("picklefile_preprocessors/target_encoder.pkl")

            # Save model locally with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            local_model_path = f"models/xgb_model_{timestamp}.pkl"
            joblib.dump(model, local_model_path)
            
        
        with mlflow.start_run(run_name="H2OAutoML", nested=True):
            mlflow.log_param("model_type", "H2OAutoML")
            mlflow.log_param("train_size", len(x_train))
            mlflow.log_metric("rmse", rmse_h2o)
            mlflow.log_metric("r2_score", r2_h2o)

            mlflow.log_artifact(h2o_model_path)
        
            mlflow.log_artifact("picklefile_preprocessors/ohe_encoder.pkl")
            mlflow.log_artifact("picklefile_preprocessors/target_encoder.pkl")
        
        
        
    # Compare R²
    if r2 > r2_h2o:
        print("XGBoost is the best model.")
        best_model_type = "XGBoost"
        best_model_file = "models/best_xgb_model.pkl"
        
        # Clean or create the best_model directory
        if os.path.exists("best_model"):
            shutil.rmtree("best_model")
        os.makedirs("best_model", exist_ok=True)
        
        # Save best model
        shutil.copy(best_model_file, "best_model/best_model.pkl")
    elif r2 < r2_h2o:
        print("H2O AutoML is the best model.")
        best_model_type = "H2OAutoML"
        best_model_dir = h2o_model_path  # full directory path returned by h2o.save_model()

        # Clear and copy H2O model folder to best_model/
        if os.path.exists("best_model"):
            shutil.rmtree("best_model")
            
        shutil.copytree(best_model_dir, "best_model")
    else:
        print("Both models have equal R². Choosing XGBoost by default.")
        best_model_type = "XGBoost"
    mlflow.set_tag("selected_model", best_model_type)
    # Save metadata
    metadata = {
        "best_model": best_model_type,
        "r2_score": max(r2, r2_h2o),
        "timestamp": timestamp
    }
    with open("best_model/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

        