# Used-Car-Price-Prediction-System

Dockerfile (optional while using wsl)
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    swig \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libomp-dev \
    libffi-dev \
    libxml2-dev \
    git \
    libgl1 \
    libgl1-mesa-glx \
    default-jre \
    && rm -rf /var/lib/apt/lists/*


Installing H2o automl

Install all the dependencies
- pip install requests
- pip install tabulate

# The following command removes the H2O module for Python.
pip uninstall h2o

# Next, use pip to install this version of the H2O Python module.
pip install https://h2o-release.s3.amazonaws.com/h2o/rel-3.46.0/7/Python/h2o-3.46.0.7-py2.py3-none-any.whl

# Add this to import csv to MySQL
- name: Import CSV into MySQL
        run: |
          mysql -h127.0.0.1 -uroot -p${{ secrets.MYSQL_ROOT_PASSWORD }} used_car -e "
          LOAD DATA LOCAL INFILE 'data/used_car_price_prediction.csv'
          INTO TABLE car_data
          FIELDS TERMINATED BY ','
          IGNORE 1 LINES;"