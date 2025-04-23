"""
Module: feature_pipeline

This module defines the logic for managing feature engineering workflows as part of
a feature store pipeline. It integrates feature preprocessing, registration,
materialization, training, and inference operations into a cohesive pipeline.
"""
import pandas as pd
from feathr import FeathrClient

from configs.conf import NUM_DATE_EVAL
from featurestore.base.utils.logger import logger
from featurestore.pipeline.infer_pipeline import InferPipeline
from featurestore.pipeline.materialize_pipeline import MaterializePipeline
from featurestore.pipeline.training_pipeline import TrainingPipeline
from featurestore.preprocess.feature_preprocessing import (
    ABUserFeaturePreprocessing,
    UserFeaturePreprocessing,
)
from featurestore.preprocess.interaction_feature_preprocessing import (
    InteractedFeaturePreprocessing,
)
from featurestore.preprocess.item_feature_preprocessing import (
    ContentFeaturePreprocessing,
)
from featurestore.preprocess.online_feature_preprocessing import (
    OnlineItemFeaturePreprocessing,
    OnlineUserFeaturePreprocessing,
)
from featurestore.registry.feature_registry import FeatureRegistry


class FeaturePipeline:
    """
    FeaturePipeline class contains various stages of the feature store pipeline,
    such as preprocessing, registration, training, materialization, and inference.

    Attributes:
        raw_data_path (str): Path to the raw data used in feature processing.
        feathr_workspace_folder (str): Path to the Feathr workspace folder.
        feature_registry_config_path (str): Path to the feature registry
            configuration file.
        training_pipeline_config_path (str): Path to the training pipeline
            configuration file.
        materialize_pipeline_config_path (str): Path to the materialization pipeline
            configuration file.
        infer_pipeline_config_path (str): Path to the inference pipeline
            configuration file.
        user_id_df (pandas.DataFrame): DataFrame containing user and item IDs
            for inferencing.
        process_lib (str): Library to use for processing.
    """

    def __init__(
        self,
        raw_data_path: str,
        infer_date: str,
        feathr_workspace_folder: str = "",
        feature_registry_config_path: str = "",
        training_pipeline_config_path: str = "",
        materialize_pipeline_config_path: str = "",
        infer_pipeline_config_path: str = "",
        user_id_df=pd.DataFrame(),
        process_lib: str = "pandas",
        spark_config: dict = None,
        job_retry: int = 10,
        job_retry_sec: int = 60,
    ):
        self.raw_data_path = raw_data_path
        self.infer_date = infer_date
        print("====================== infer_date:", self.infer_date)
        if feathr_workspace_folder == "":
            self.client = None
        else:
            self.client = FeathrClient(
                feathr_workspace_folder,
                job_retry=job_retry,
                job_retry_sec=job_retry_sec,
            )
        self.feature_registry_config_path = feature_registry_config_path
        self.training_pipeline_config_path = training_pipeline_config_path
        self.materialize_pipeline_config_path = materialize_pipeline_config_path
        self.infer_pipeline_config_path = infer_pipeline_config_path
        self.user_id_df = user_id_df
        self.process_lib = process_lib
        self.spark_config = spark_config

    def preprocess_features(self):
        """
        Preprocesses raw data to generate features required for different categories.
        Utilizes modular preprocessing implementations for each specific feature type.
        """
        UserFeaturePreprocessing(
            self.process_lib, self.raw_data_path, spark_config=self.spark_config
        ).run()
        ABUserFeaturePreprocessing(
            self.process_lib, self.raw_data_path, spark_config=self.spark_config
        ).run()
        ContentFeaturePreprocessing(
            self.process_lib, self.raw_data_path, spark_config=self.spark_config
        ).run()
        InteractedFeaturePreprocessing(
            self.process_lib, self.raw_data_path, spark_config=self.spark_config
        ).run()
        OnlineItemFeaturePreprocessing(
            self.process_lib, self.raw_data_path, spark_config=self.spark_config
        ).run()
        OnlineUserFeaturePreprocessing(
            self.process_lib, self.raw_data_path, spark_config=self.spark_config
        ).run()

    def register_features(self):
        """
        Registers feature definitions into the feature registry.
        """
        FeatureRegistry(
            config_path=self.feature_registry_config_path,
            feathr_client=self.client,
            raw_data_path=self.raw_data_path,
        ).run()

    def get_features_for_training_pipeline(self):
        """
        Fetches and processes features required for training a machine learning model.
        """
        TrainingPipeline(
            config_path=self.training_pipeline_config_path,
            feathr_client=self.client,
            raw_data_path=self.raw_data_path,
            infer_date=self.infer_date,
        ).run()

    def materialize_online_features(self):
        """
        Materializes features for online serving.
        """
        MaterializePipeline(
            config_path=self.materialize_pipeline_config_path,
            feathr_client=self.client,
            raw_data_path=self.raw_data_path,
            infer_date=self.infer_date,
        ).run()

    def materialize_offline_features(self):
        """
        Materializes features for offline evaluations or batch processing.
        """
        MaterializePipeline(
            config_path=self.materialize_pipeline_config_path,
            feathr_client=self.client,
            materialize_for_eval=True,
            raw_data_path=self.raw_data_path,
            infer_date=self.infer_date,
            num_date_eval=NUM_DATE_EVAL,
        ).run()

    def get_features_for_infer_pipeline(self):
        """
        Fetches features for inferencing tasks.
        """
        InferPipeline(
            feathr_client=self.client,
            user_item_df=self.user_id_df,
            config_path=self.infer_pipeline_config_path,
        ).run()

    def run_all(self):
        """
        Executes the entire feature pipeline workflow.

        This method runs all stages of the feature pipeline in sequence, including:
            - Preprocessing features.
            - Registering features.
            - Retrieving features for the training pipeline.
            - Materializing features for online and offline use.
        """
        logger.info("PREPROCESS FEATURES")
        self.preprocess_features()
        import gc

        gc.collect()
        logger.info("REGISTER FEATURES")
        self.register_features()
        logger.info("GET TRAINING FEATURES")
        self.get_features_for_training_pipeline()
        logger.info("MATERIALIZE ONLINE FEATURES")
        self.materialize_online_features()
        logger.info("MATERIALIZE OFFLINE FEATURES")
        self.materialize_offline_features()
