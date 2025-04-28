"""
Module: training_pipeline

This module provides the `TrainingPipeline` class, which is responsible for generating
training datasets by querying and materializing offline features from a feature store.
It prepares the data in a format ready for model training, utilizing configurations
to streamline the feature retrieval process.
"""
import shutil
from glob import glob
from pathlib import Path
from typing import List

from feathr import FeatureQuery, ObservationSettings
from feathr.utils.job_utils import get_result_df

import configs.conf
from configs.conf import TIMESTAMP_COLUMN, TIMESTAMP_FORMAT
from featurestore.base.schemas.pipeline_config import TrainingPipelineConfig
from featurestore.base.utils.config import parse_training_config
from featurestore.base.utils.spark import SparkOperations
from featurestore.base.utils.utils import return_or_load
from featurestore.constants import DataName
from featurestore.daily_data_utils import get_date_before
from featurestore.transform.key_definition import KeyDefinition


class TrainingPipeline:
    """
    TrainingPipeline is responsible for producing training datasets by querying
    and materializing offline features based on configurations, and preparing
    them for model training.

    Attributes:
        training_config (TrainingPipelineConfig): Parsed configuration object
        client: Feathr client for interacting with the feature store.
        raw_data_path (Path): Path to the base raw data directory.
        timestamp_column (str): Column name for event timestamps in the observation data
        timestamp_format (str): Format of the timestamp used in the observation data.
        observation_source_path (Path): Path to the observation source data.
        output_path (Path): Path where the resulting offline features are saved.
        key_collection (dict): Collection of key definitions for query construction.
        feature_query (List[str]): List of feature queries used to retrieve
        offline features.
    """

    def __init__(
        self,
        config_path,
        feathr_client,
        raw_data_path: str,
        infer_date,
        timestamp_column: str = TIMESTAMP_COLUMN,
        timestamp_format: str = TIMESTAMP_FORMAT,
    ):
        self.training_config = return_or_load(
            config_path, TrainingPipelineConfig, parse_training_config
        )
        self.client = feathr_client
        self.raw_data_path = Path(
            raw_data_path + "/" + self.training_config.raw_data_path
        )
        self.timestamp_column = timestamp_column
        self.timestamp_format = timestamp_format
        self.for_date = infer_date

        self.is_init_df = self._check_output_data_path()
        if self.is_init_df:
            self.observation_source_path = (
                self.raw_data_path / f"{DataName.OBSERVATION_FEATURES}.parquet"
            )
            self.output_path = (
                self.raw_data_path / "features" / f"{DataName.OFFLINE_FEATURES}.parquet"
            )
        else:
            self.for_date = get_date_before(int(self.for_date), num_days_before=1)
            self.observation_source_path = (
                self.raw_data_path
                / f"{DataName.OBSERVATION_FEATURES}.parquet"
                / f"{configs.conf.FILENAME_DATE_COL}={self.for_date}"
            )
            self.output_path = (
                self.raw_data_path
                / "features"
                / f"{DataName.OFFLINE_FEATURES}.parquet"
                / f"{configs.conf.FILENAME_DATE_COL}={self.for_date}"
            )

        self.key_collection = KeyDefinition().key_collection
        self.feature_query: List[str] = []

    def _check_output_data_path(self):
        save_output_path = (
            self.raw_data_path / "features" / f"{DataName.OFFLINE_FEATURES}.parquet"
        )
        if save_output_path.exists():
            subfolder_list = glob(
                str(save_output_path) + f"/{configs.conf.FILENAME_DATE_COL}=*"
            )
            if len(subfolder_list) > 0:
                return False
        return True

    def query_feature(self):
        """
        Prepares feature queries for offline feature retrieval.

        For each feature query specified in the configuration, this method constructs a
        `FeatureQuery` object using the feature names and the corresponding key
        definitions
        """
        print(self.training_config.feature_queries)
        for feature_query in self.training_config.feature_queries:
            query = FeatureQuery(
                feature_list=feature_query.feature_names,
                key=[self.key_collection[i] for i in feature_query.key_names],
            )
            print(query)

            self.feature_query.append(query)

    def _get_offline_features(self):
        """
        Retrieves offline features from observation data and saves the results
        to an output file.

        This method uses the Feathr client to retrieve the desired features based on
        the prepared `feature_query` list. The observation data and retrieval
        configurations are used to save the results in a specified output path.

        Returns:
            None: The results are saved to the `output_path` defined in the pipeline.
        """
        settings = ObservationSettings(
            observation_path=str(self.observation_source_path),
            event_timestamp_column=self.timestamp_column,
            timestamp_format=self.timestamp_format,
        )
        self.client.get_offline_features(
            observation_settings=settings,
            feature_query=self.feature_query,
            output_path=str(self.output_path),
            execution_configurations=self.training_config.spark_execution_config,
        )
        self.client.wait_job_to_finish(timeout_sec=100000)

    def _get_result_df(self):
        """
        Loads the resulting offline feature dataset into a pandas DataFrame.

        This method reads the saved output dataset of the offline features
        from the output path and returns it as a pandas DataFrame.

        Returns:
            pandas.DataFrame: The dataset containing the retrieved offline features.
        """
        result_df = get_result_df(self.client, res_url=str(self.output_path))
        return result_df

    def _repartition_offline_df(self, delete_init_data=False):
        """
        Repartition the initialized offline feature dataframe and save to a new path.
        Only use when process offline feature for the first time (is_init_df=True)
        """
        new_output_path = (
            self.raw_data_path / "features" / f"{DataName.OFFLINE_FEATURES}.parquet"
        )
        # df = self._get_result_df()
        spark = SparkOperations().get_spark_session()
        df = spark.read.option("header", True).parquet(str(self.output_path))
        df.write.option("header", True).partitionBy(
            configs.conf.FILENAME_DATE_COL
        ).mode("overwrite").parquet(str(new_output_path))
        if delete_init_data:
            self._delete_init_offline_folder()

    def _delete_init_offline_folder(self):
        """
        Delete initialized offline feature dataframe.
        Only use when process offline feature for the first time (is_init_df=True)
        """
        if self.output_path.exists():
            shutil.rmtree(self.output_path)

    def run(self):
        """
        Executes the entire training pipeline workflow.
        """
        self.query_feature()
        self._get_offline_features()
