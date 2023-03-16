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

- Train period - (1919-1924) -- 2015
- Test period  - 2016 -- 2022
- Predict period - 2023

## Model 

-- Prophet

## Results 
-- Directory data/08_reporting

### MEAN ABSOLUTE ERROR(MAE) ON TEST PERIOD

![mae_and_mape](https://user-images.githubusercontent.com/23450113/225747086-aad405c3-7536-4fce-bde1-ed624c7e7a37.png)


### MEAN ABSOLUTE PERCENTAGE ERROR(MAPE) ON TEST PERIOD 

![mape](https://user-images.githubusercontent.com/23450113/225747114-3d4e6df3-67fe-4c49-bd38-61a5118bad0d.png)


### PREDICTION 2023 ACORSS ALL STATES

![prediction_and_mae](https://user-images.githubusercontent.com/23450113/225747185-556c5fea-64de-4db9-a122-2a06ea2cd838.png)
