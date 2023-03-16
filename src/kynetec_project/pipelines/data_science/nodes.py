from typing import Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


def data_preprocessing(df: pd.DataFrame, train_target_variable: Dict) -> pd.DataFrame:
    """
    Preprocessing data in a training format for the model.
    Args:
        df: A pandas dataframe.
        train_target_variable: Columns and parameters used in pre-processing data.

    Returns:
        A preprocessed dataframe.
    """

    date_column = train_target_variable['date_column']
    target_column = train_target_variable['target_column']
    other_column = train_target_variable['other_columns'][0]

    # Filter only relevant data
    df = df[[date_column, other_column, target_column]]
    
    # Format data column for Prophet inputs
    df[date_column] = pd.to_datetime(df[date_column].astype(str), format='%Y')
    df = df.rename(columns={date_column: "ds", target_column: "y"})
    return df


def train_predict_single_group(group: pd.DataFrame, train_target_variable,
                               training_hyperparameters:Dict) -> pd.DataFrame:

    """
    Training and predicting a time series.
    Args:
        group: A time series pandas dataframe.
        train_target_variable: Columns and parameters used in training and testing.
        training_hyperparameters: Hyper parameters used in training.

    Returns:
        A time series dataframe with historical and future predictions
    """
    # Train Test variables and parameters
    forecast_period = train_target_variable['forecast_period']
    split_date = train_target_variable['split_date']
    other_column = train_target_variable['other_columns']

    seasonality_mode = training_hyperparameters['seasonality_mode']
    n_changepoints = training_hyperparameters['n_changepoints']
    changepoint_prior_scale = training_hyperparameters['changepoint_prior_scale']

    # Initiate the model
    # =============== PROPHET ======================
    m = Prophet(seasonality_mode=seasonality_mode,
                n_changepoints=n_changepoints,
                changepoint_prior_scale=changepoint_prior_scale)
    # ================ PROPHET ==================

    # Train and test split
    group_train = group[group["ds"] <= split_date]
    group_test = group[group["ds"] > split_date]
    # Fit the model
    m.fit(group_train)
    # Exclude train predictions
    future = m.make_future_dataframe(periods=forecast_period + len(group_test), freq='YS', include_history=False)
    # Make predictions
    forecast = m.predict(future)[['ds', 'yhat']]
    forecast[other_column] = group_test[other_column].iloc[0]

    return forecast


def train_make_predictions(df: pd.DataFrame, train_target_variable: Dict,
                     training_hyperparameters: Dict) -> pd.DataFrame:
    """
    Training and predictions for multiple time series.

    Args:
        df: A time series pandas dataframe.
        train_target_variable: Columns and parameters used in pre-processing data.
        training_hyperparameters: Hyper parameters used in training.

    Returns:
        A dataframe with historical and future predictions for multiple time series.

    """

    # Get the grouping column
    other_column = train_target_variable['other_columns'][0]
    unique_tickers = df[other_column].unique()
    # Each group i.e. state
    groups_by_ticker = df.groupby(other_column)
    ticker_list = unique_tickers.tolist()
    # ticker_list = ticker_list[1:5]                   # UNCOMMENT WHEN PREDICTING FOR FIRST FEW STATES
    all_states_forecast = pd.DataFrame()

    # Loop through each ticker(state)
    for ticker in ticker_list:
        # Get the data for the ticker
        group = groups_by_ticker.get_group(ticker)
        group = group.sort_values(by=['ds'])
        # Make forecast
        forecast = train_predict_single_group(group, train_target_variable, training_hyperparameters)
        # Add the forecast results to the dataframe
        all_states_forecast = pd.concat((all_states_forecast, forecast))
    return all_states_forecast


def report_mae_mape(df_actual: pd.DataFrame, df_hist_future_forecast: pd.DataFrame,
                    train_target_variable: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Calculates Mean absolute error and Mean absolute percentage error.
    Args:
        df_actual: A dataframe with actual historical values (y_true).
        df_hist_future_forecast: A dataframe with the historical and future predicted values for all time series.
        train_target_variable: Columns and parameters used in data pre-processing and predictions.

    Returns:
        Tuple of dataframes with info
        comparing predicted and actual, future forecasts and model performance metrics.

    """
    other_column = train_target_variable['other_columns'][0]
    last_historical_date = max(df_actual['ds'])
    # Extract future forecast
    future_forecast = df_hist_future_forecast[df_hist_future_forecast["ds"] > last_historical_date]

    # Filter out future forecast dates
    real_and_forecast = pd.merge(left=df_hist_future_forecast, right=df_actual,
                                 on=["ds", other_column], how="inner")
    # Difference between prediction and forecast
    real_and_forecast["residual"] = abs(real_and_forecast["y"] - real_and_forecast["yhat"])
    real_and_forecast["percentage_error"] = (real_and_forecast["residual"] / real_and_forecast["y"]) * 100

    # Calculate MAE, MAPE
    mae_mape = real_and_forecast.groupby(other_column)\
        .agg({'residual': ['mean'],
              'percentage_error': ['mean']})

    mae_mape.columns = ['mae', 'mape']
    mae_mape = mae_mape.reset_index()

    return real_and_forecast, future_forecast, mae_mape


def plot_predictions(future_forecast: pd.DataFrame, train_target_variable: Dict) -> plt:

    """
    Plots predictions for multiple time series given the specified period.
    Args:
        future_forecast: A pandas dataframe with future predictions for multiple time series.
        train_target_variable: Columns and parameters used in data pre-processing and predictions.

    Returns:
        Plot with future predictions for multiple time series.

    """
    plt.style.use('_mpl-gallery')

    other_column = train_target_variable['other_columns'][0]
    # Future forecast for all states
    x = future_forecast[other_column]
    y = future_forecast['yhat']

    # plot:
    plt.rcParams["figure.figsize"] = [12, 12]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(x, y, 'o', ms=12, color='darkorange')
    plt.xticks(rotation=45, fontsize=12, fontweight='semibold')
    plt.yticks(fontsize=13, fontweight='semibold')
    plt.xlabel(other_column)
    plt.ylabel('PREDICTIONS 2023')

    plt.title("CORN Predictions 2023")
    return plt


def plot_mae_mape(mae_mape: pd.DataFrame,  train_target_variable: Dict) -> plt:

    """
    Plots the historical MAE and MAPE for all time series.
    Args:
        mae_mape: A pandas data frame metrics of mae and mape.
        train_target_variable: Parameters used in data pre-processing and predictions.

    Returns:
        A plot with MAE and MAPE on test set for all time series.
    """

    other_column = train_target_variable['other_columns'][0]
    mae_mape = mae_mape.set_index(other_column)
    plt.figure(figsize=(100, 100))
    mae_mape.plot.bar(align='center')
    plt.xticks(rotation=45, fontsize=5, fontweight='semibold')
    plt.yticks(fontsize=5, fontweight='semibold')
    for i in range(len(mae_mape)):
        plt.text(i-0.15, mae_mape['mae'][i] + 50, str(round(mae_mape['mape'][i], 2)) + "%", ha='center',
                 c='black', fontsize=2.8, fontweight='heavy')

    plt.title(f"MAE and MAPE on TEST SET, PERIOD > {train_target_variable['split_date']}")
    plt.legend(loc="upper right", fontsize='x-small')

    return plt


def plot_mape(mae_mape: pd.DataFrame,  train_target_variable: Dict) -> plt:

    """
    Plots the historical MAE and MAPE for all time series.
    Args:
        mae_mape: A pandas data frame metrics of mae and mape.
        train_target_variable: Parameters used in data pre-processing and predictions.

    Returns:
        A plot with MAPE on Test set for all time series.
    """

    other_column = train_target_variable['other_columns'][0]

    # MAPE to compare across states
    mape = mae_mape[[other_column, 'mape']]
    mape = mape.set_index(other_column)
    plt.figure(figsize=(100, 100))
    mape.plot.bar(align='center')
    plt.xticks(rotation=45, fontsize=5, fontweight='semibold')
    plt.yticks(fontsize=5, fontweight='semibold')
    plt.ylabel('MAPE percentage')

    plt.title(f"MAPE on TEST SET, PERIOD > {train_target_variable['split_date']}")
    plt.legend(loc="upper right", fontsize='x-small')

    return plt
