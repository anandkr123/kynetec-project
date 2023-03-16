from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import load_dataset, filter_dataset, fix_datatypes


def create_pipeline(**kwargs) -> Pipeline:
    """
    Constructs a pipeline of nodes.
    Args:
        **kwargs:

    Returns:
        Pipeline executing specified nodes sequentially.
    """
    return pipeline(
        [
            # node(                                         # Comment this node after first run.
            #     func=load_dataset,                        # May exceed RAM IF run every time
            #     inputs="corn_parquet",
            #     outputs="corn",
            #     name="read_corn_parquet_object"
            # ),
            node(
                func=filter_dataset,
                inputs=["corn", "params:corn_filters"],
                outputs="corn_filtered",
                name="filter_corn_data"
            ),
            node(
                func=fix_datatypes,
                inputs=["corn_filtered", "params:corn_fix_datatypes"],
                outputs="corn_fixed_datatypes",
                name="fix_datatypes_corn_data"
            )
        ]
    )
