"""
Module: pipeline_config

This module defines configuration schemas for various types of data pipelines, such as
training, materialization, and inference pipelines. These schemas use Pydantic models
to provide strongly-typed and validated structures that ensure consistent and reliable
pipeline setups.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel


class FeatureTableConfig(BaseModel):
    """
    Configuration for a feature table in a pipeline.

    Attributes:
        feature_table_names (List[str]): A list of names of the feature tables.
        setting_name (str): An optional context or tag for the configuration.
        key_names (List[str]): The key column names used as unique identifiers
        feature_names (List[str]): The names of features present in the table.
    """

    feature_table_names: List[str]
    setting_name: str = ""
    key_names: List[str]
    feature_names: List[str]


class FeatureQueryConfig(BaseModel):
    """
    Configuration for querying features from a feature table.

    Attributes:
        feature_names (List[str]): The list of feature names to query.
        key_names (List[str]): The key column names representing unique identifiers
            for data retrieval.
    """

    feature_names: List[str]
    key_names: List[str]


class TrainingPipelineConfig(BaseModel):
    """
    Configuration for a training pipeline.

    Attributes:
        raw_data_path (str): The file path to the raw data used in training.
        feature_queries (List[FeatureQueryConfig]): A list of configurations defining
            the features to query for training.
        spark_execution_config (Dict): A dictionary of Spark execution configurations
            for managing distributed processing during training.
    """

    raw_data_path: str
    feature_queries: List[FeatureQueryConfig]
    spark_execution_config: Dict


class MaterializePipelineConfig(BaseModel):
    """
    Configuration for a materialization pipeline.

    Attributes:
        feature_tables (List[FeatureTableConfig]): A list of configurations for the
            feature tables to materialize.
        online_spark_execution_configs (Optional[Dict]): Spark execution configurations
            for online storage.
        offline_spark_execution_configs (Optional[Dict]): Spark execution configurations
            for offline storage.
        save_dir_path (str): The directory path to save materialized features.
    """

    feature_tables: List[FeatureTableConfig]
    online_spark_execution_configs: Optional[Dict] = None
    offline_spark_execution_configs: Optional[Dict] = None
    save_dir_path: str = ""


class InferPipelineConfig(BaseModel):
    """
    Configuration for an inference pipeline.

    Attributes:
        feature_tables (List[FeatureTableConfig]): A list of feature table
            configurations used for inference.
    """

    feature_tables: List[FeatureTableConfig]
