# Kynetec-project

## How to install dependencies

To install them, run:

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

## Train Test period 

- Train period -    (1919-1924) - 2015
- Test period  -    2016 - 2022
- Predict period -  2023

## Model 

```
Prophet time series forecaster [Read more](https://facebook.github.io/prophet/docs/quick_start.html)
```

## Directory structure  

### Directory data/01_raw

#### File explanations 

- Parquet object

### Directory data/02_intermediate

#### File explanations 

- corn.csv              raw corn file from parquet obj.  

-Directory data/03_primary

### File explanations 

- corn_filtered.csv      corn data after columns and values filttering.
- corn_final.csv         final corn data after type casting

### Directory data/04_feature

#### File explanations 

- corn.csv data preprocessed for model input format.

### Directory data/07_model_output

#### File explanations 

- future_forecast.csv            only future forecast for all states.
- hist_future_forecasts.csv      forecast for test and future period across all states.

### Directory data/08_reporting

#### File explanations 
- mae_and_mape.png           plot MEAN ABSOLUTE ERROR and MEAN ABSOLUTE PERCENTAGE ERROR on Test data (PERIOD 2016-2022) acrocss all states.
- mae_mape.csv               MEAN ABSOLUTE ERROR and MEAN ABSOLUTE PERCENTAGE ERROR on Test data (PERIOD 2016-2022) acrocss all states.
- mape.png                   MEAN ABSOLUTE PERCENTAGE ERROR for relative comparison across all states.
- prediction_and_mae.png     Predictions 2023 for all states.
- real_and_forecast.csv      Historical and future predictions along with deviations AND percentage deviations for each year across all states.

### Directory src/tests
- Test caases for data transformations

## Results

### MEAN ABSOLUTE ERROR(MAE) ON TEST PERIOD

![mae_and_mape](https://user-images.githubusercontent.com/23450113/225747086-aad405c3-7536-4fce-bde1-ed624c7e7a37.png)


### MEAN ABSOLUTE PERCENTAGE ERROR(MAPE) ON TEST PERIOD 

![mape](https://user-images.githubusercontent.com/23450113/225747114-3d4e6df3-67fe-4c49-bd38-61a5118bad0d.png)


### PREDICTION 2023 ACORSS ALL STATES

![prediction_and_mae](https://user-images.githubusercontent.com/23450113/225747185-556c5fea-64de-4db9-a122-2a06ea2cd838.png)
