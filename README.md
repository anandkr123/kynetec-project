# Time-series corn Predictions 2023

Download project and mark **src** as Source Root by right click on it.
Create and activate virtual environment in PyCharm. Then migrate to src directory.

## How to install dependencies

To install them, run:

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run the Kedro project with:

```
kedro run
```

## How to test your Kedro project

You can run the tests as follows:

```
kedro test
```

## Train Test period 

- Train period -    (1919-1924) - 2015
- Test period  -    2016 - 2022
- Predict period -  2023


## Model 

- Prophet time series forecaster [Read more](https://facebook.github.io/prophet/docs/quick_start.html).


## Directory structure  

### Directory data/01_raw

#### File explanations 

- Parquet object

### Directory data/02_intermediate

#### File explanations 

- corn.csv        &nbsp;&nbsp;      raw corn file from parquet obj.  Not uploaded due to large size. For details see pipelines/data_engineering/pipeline.py (line 18)

### Directory data/03_primary

#### File explanations 

- corn_filtered.csv   &nbsp;&nbsp;   corn data after columns and values filttering.
- corn_final.csv     &nbsp;&nbsp;    final corn data after type casting

### Directory data/04_feature

#### File explanations 

- corn.csv data preprocessed for model input format.

### Directory data/07_model_output

#### File explanations 

- future_forecast.csv    &nbsp;&nbsp;        only future forecast for all states.
- hist_future_forecasts.csv  &nbsp;&nbsp;    forecast for test and future period across all states.

### Directory data/08_reporting

#### File explanations 
- mae_and_mape.png &nbsp;&nbsp;          Plot MEAN ABSOLUTE ERROR and MEAN ABSOLUTE PERCENTAGE ERROR on Test data (PERIOD 2016-2022) acrocss all states.
- mae_mape.csv      &nbsp;&nbsp;         MEAN ABSOLUTE ERROR and MEAN ABSOLUTE PERCENTAGE ERROR on Test data (PERIOD 2016-2022) acrocss all states.
- mape.png           &nbsp;&nbsp;        MEAN ABSOLUTE PERCENTAGE ERROR for relative comparison across all states.
- future_predictions.png  &nbsp;&nbsp;   Predictions 2023 for all states.
- real_and_forecast.csv   &nbsp;&nbsp;   Real and forecasts on test set along with residuals AND percentage deviations for each year across all states.

### Directory src/tests
- Test cases for data transformations

## Results

### MEAN ABSOLUTE ERROR(MAE) ON TEST PERIOD

![mae_and_mape](https://user-images.githubusercontent.com/23450113/225747086-aad405c3-7536-4fce-bde1-ed624c7e7a37.png)


### MEAN ABSOLUTE PERCENTAGE ERROR(MAPE) ON TEST PERIOD 

![mape](https://user-images.githubusercontent.com/23450113/225747114-3d4e6df3-67fe-4c49-bd38-61a5118bad0d.png)


### PREDICTION 2023 ACORSS ALL STATES

![future_predictions](https://user-images.githubusercontent.com/23450113/225773553-122d1713-d359-42f6-b7fe-3f2b23effc05.png)



