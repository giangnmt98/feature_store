import shutil

import pandas as pd
import pytest

from featurestore.base.utils.logger import logger
from featurestore.feature_pipeline import FeaturePipeline


class TestFeaturePipeline:
    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        test_dir = request.fspath.dirname
        monkeypatch.chdir(request.fspath.dirname)
        return test_dir

    def test_feature_pipeline_pyspark(self):
        self.clean_folder()
        logger.info("PREPROCESS FEATURES")
        FeaturePipeline(
            raw_data_path="data/processed/",
            process_lib="pyspark",
        ).preprocess_features()

        user_item_df = pd.read_parquet(
            "data/processed/preprocessed_features/observation_features.parquet"
        )
        user_item_df = user_item_df[["user_id", "item_id"]].drop_duplicates()[:5]
        new_user_dict = {"user_id": ["xxxxxxxxxx"], "item_id": ["XXXXXXXXXX"]}
        new_user = pd.DataFrame(data=new_user_dict)
        user_item_df = pd.concat([user_item_df, new_user])

        pipeline = FeaturePipeline(
            raw_data_path="data/processed/",
            feathr_workspace_folder="configs/feathr_config.yaml",
            feature_registry_config_path="configs/feature_registry_config.yaml",
            training_pipeline_config_path="configs/training_pipeline_config.yaml",
            materialize_pipeline_config_path="configs/materialize_pipeline_config.yaml",
            infer_pipeline_config_path="configs/infer_pipeline_config.yaml",
            user_id_df=user_item_df,
            process_lib="pyspark",
        )
        logger.info("REGISTER FEATURES")
        pipeline.register_features()
        logger.info("GET TRAINING FEATURES")
        pipeline.get_features_for_training_pipeline()
        logger.info("MATERIALIZE OFFLINE FEATURES")
        pipeline.materialize_offline_features()
        logger.info("MATERIALIZE ONLINE FEATURES")
        pipeline.materialize_online_features()
        logger.info("GET INFERRING FEATURES")
        pipeline.get_features_for_infer_pipeline()

    def clean_folder(self):
        paths = [
            "experiments/",
            "test_experiments/",
            "lightning_logs/",
            "data/processed/preprocessed_features/",
            "*.log",
        ]
        for f in paths:
            shutil.rmtree(f, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup a testing directory once we are finished."""

    def remove_test_dir():
        paths = [
            "experiments/",
            "test_experiments/",
            "lightning_logs/",
            "data/processed/preprocessed_features/",
            "*.log",
            "debug/",
        ]
        import os

        for f in paths:
            shutil.rmtree(
                f"{os.path.dirname(os.path.realpath(__file__))}/{f}", ignore_errors=True
            )

    request.addfinalizer(remove_test_dir)
