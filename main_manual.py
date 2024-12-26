import logging
import sys
import time
import warnings

import typer

from configs import conf
from featurestore.feature_pipeline import FeaturePipeline

warnings.filterwarnings("ignore")
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def run_pipeline(
    process_lib: str = typer.Option("pyspark", help="Process library to use."),
    method: str = "run_all",
):
    pipeline = FeaturePipeline(
        raw_data_path="data/processed/",
        feathr_workspace_folder="configs/feathr_config.yaml",
        feature_registry_config_path="configs/feature_registry_config.yaml",
        training_pipeline_config_path="configs/training_pipeline_config.yaml",
        materialize_pipeline_config_path="configs/materialize_pipeline_config.yaml",
        infer_pipeline_config_path="configs/infer_pipeline_config.yaml",
        process_lib=process_lib,
    )
    getattr(pipeline, method)()


def is_debugging() -> bool:
    return (gettrace := getattr(sys, "gettrace")) and gettrace()


@app.command()
def test_func():
    pass


if __name__ == "__main__":
    if not is_debugging():
        app()
    else:
        FeaturePipeline(
            raw_data_path="data/processed/",
            feathr_workspace_folder="configs/feathr_config.yaml",
            feature_registry_config_path="configs/feature_registry_config.yaml",
            training_pipeline_config_path="configs/training_pipeline_config.yaml",
            materialize_pipeline_config_path="configs/materialize_pipeline_config.yaml",
            infer_pipeline_config_path="configs/infer_pipeline_config.yaml",
            process_lib="pandas",
        ).run_all()
