"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

from kynetec_project.pipelines import data_engineering as de
from kynetec_project.pipelines import data_science as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_engineering_pipeline = de.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    return {
        "__default__": data_engineering_pipeline + data_science_pipeline,
        "de": data_engineering_pipeline,
        "ds": data_science_pipeline
    }

