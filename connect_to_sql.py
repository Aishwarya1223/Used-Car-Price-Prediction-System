import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Read CSV
df = pd.read_csv("new_data/new_data.csv")
df=df.tail(10)


# Build DB connection string from .env
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
host = os.getenv("MYSQL_HOST")
port = os.getenv("MYSQL_PORT")
db = os.getenv("MYSQL_DB")

engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db}")

# Write to MySQL
df.to_sql(
    name="car_data",
    con=engine,
    if_exists="append",
    index=False,
    dtype={
        "model": sqlalchemy.types.String(50),
        "year": sqlalchemy.types.Integer(),
        "price": sqlalchemy.types.Float(),
        "transmission": sqlalchemy.types.String(20),
        "mileage": sqlalchemy.types.Integer(),
        "fuelType": sqlalchemy.types.String(20),
        "tax": sqlalchemy.types.Float(),
        "mpg": sqlalchemy.types.Float(),
        "engineSize": sqlalchemy.types.Float(),
        "brand": sqlalchemy.types.String(50)
    }
)

print("The table was created and the records were inserted successfully..")
