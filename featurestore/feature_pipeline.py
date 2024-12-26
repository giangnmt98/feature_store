import pandas as pd
from feathr import FeathrClient

from featurestore.base.utils.logger import logger
from featurestore.feature_preprocessing import (
    ABUserFeaturePreprocessing,
    ContentFeaturePreprocessing,
    InteractedFeaturePreprocessing,
    OnlineItemFeaturePreprocessing,
    OnlineUserFeaturePreprocessing,
    UserFeaturePreprocessing,
)
from featurestore.pipeline.infer_pipeline import InferPipeline
from featurestore.pipeline.materialize_pipeline import MaterializePipeline
from featurestore.pipeline.training_pipeline import TrainingPipeline
from featurestore.registry.feature_registry import FeatureRegistry


class FeaturePipeline:
    def __init__(
        self,
        raw_data_path: str,
        feathr_workspace_folder: str = "",
        feature_registry_config_path: str = "",
        training_pipeline_config_path: str = "",
        materialize_pipeline_config_path: str = "",
        infer_pipeline_config_path: str = "",
        user_id_df=pd.DataFrame(),
        process_lib: str = "pandas",
    ):
        self.raw_data_path = raw_data_path
        if feathr_workspace_folder == "":
            self.client = None
        else:
            self.client = FeathrClient(feathr_workspace_folder)
        self.feature_registry_config_path = feature_registry_config_path
        self.training_pipeline_config_path = training_pipeline_config_path
        self.materialize_pipeline_config_path = materialize_pipeline_config_path
        self.infer_pipeline_config_path = infer_pipeline_config_path
        self.user_id_df = user_id_df
        self.process_lib = process_lib

    def preprocess_features(self):
        UserFeaturePreprocessing(self.process_lib, self.raw_data_path).run()
        ABUserFeaturePreprocessing(self.process_lib, self.raw_data_path).run()
        ContentFeaturePreprocessing(self.process_lib, self.raw_data_path).run()
        InteractedFeaturePreprocessing(self.process_lib, self.raw_data_path).run()
        OnlineItemFeaturePreprocessing(self.process_lib, self.raw_data_path).run()
        OnlineUserFeaturePreprocessing(self.process_lib, self.raw_data_path).run()

    def register_features(self):
        FeatureRegistry(
            config_path=self.feature_registry_config_path,
            feathr_client=self.client,
        ).run()

    def get_features_for_training_pipeline(self):
        TrainingPipeline(
            config_path=self.training_pipeline_config_path,
            feathr_client=self.client,
        ).run()

    def materialize_features(self):
        MaterializePipeline(
            config_path=self.materialize_pipeline_config_path,
            feathr_client=self.client,
        ).run()

    def get_features_for_infer_pipeline(self):
        InferPipeline(
            feathr_client=self.client,
            user_item_df=self.user_id_df,
            config_path=self.infer_pipeline_config_path,
        ).run()

    def run_all(self):
        logger.info("PREPROCESS FEATURES")
        self.preprocess_features()
        logger.info("REGISTER FEATURES")
        self.register_features()
        logger.info("GET TRAINING FEATURES")
        self.get_features_for_training_pipeline()
        logger.info("MATERIALIZE FEATURES")
        self.materialize_features()
