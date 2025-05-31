use used_car;

CREATE TABLE IF NOT EXISTS car_data (
  model VARCHAR(50),
  year INT,
  price FLOAT,
  transmission VARCHAR(20),
  mileage INT,
  fuelType VARCHAR(20),
  tax FLOAT,
  mpg FLOAT,
  engineSize FLOAT,
  brand VARCHAR(50),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


select * from car_data;