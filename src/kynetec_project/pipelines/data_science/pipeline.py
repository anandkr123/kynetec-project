from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import (data_preprocessing, train_make_predictions, report_mae_mape,
                    plot_predictions, plot_mae_mape, plot_mape)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline of nodes.

    Args:
        **kwargs:

    Returns:
        Pipeline executing specified nodes sequentially.
    """
    return pipeline(
        [
            node(
                func=data_preprocessing,
                inputs=["corn_fixed_datatypes", "params:train_target_variable"],
                outputs="data_preprocessed",
                name="prepare_training_data",
            ),
            node(
                func=train_make_predictions,
                inputs=["data_preprocessed", "params:train_target_variable", "params:training_hyperparameters"],
                outputs="hist_future_forecasts",
                name="all_states_predictions",
            ),
            node(
                func=report_mae_mape,
                inputs=["data_preprocessed", "hist_future_forecasts", "params:train_target_variable"],
                outputs=["real_and_forecast", "future_forecast", "mae_mape"],
                name="report_mae_mape"
            ),
            node(
                func=plot_predictions,
                inputs=["future_forecast", "params:train_target_variable"],
                outputs="plot_predictions",
                name="plot_predictions",
            ),
            node(
                func=plot_mae_mape,
                inputs=["mae_mape", "params:train_target_variable"],
                outputs="plot_mae_and_mape",
                name="plot_mae_and_mape"
            ),
            node(
                func=plot_mape,
                inputs=["mae_mape", "params:train_target_variable"],
                outputs="plot_mape",
                name="plot_mape"
            )
        ]
    )
