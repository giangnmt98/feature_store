from pathlib import Path
from typing import List

from feathr import FeatureQuery, ObservationSettings
from feathr.utils.job_utils import get_result_df

from configs.conf import TIMESTAMP_COLUMN, TIMESTAMP_FORMAT
from featurestore.base.schemas.pipeline_config import TrainingPipelineConfig
from featurestore.base.utils.config import parse_training_config
from featurestore.base.utils.utils import return_or_load
from featurestore.constants import DataName
from featurestore.transform.key_definition import KeyDefinition


class TrainingPipeline:
    def __init__(
        self,
        config_path,
        feathr_client,
        timestamp_column: str = TIMESTAMP_COLUMN,
        timestamp_format: str = TIMESTAMP_FORMAT,
    ):
        self.training_config = return_or_load(
            config_path, TrainingPipelineConfig, parse_training_config
        )
        self.client = feathr_client
        self.raw_data_path = Path(self.training_config.raw_data_path)
        self.timestamp_column = timestamp_column
        self.timestamp_format = timestamp_format
        self.observation_source_path = (
            self.raw_data_path / f"{DataName.OBSERVATION_FEATURES}.parquet"
        )
        self.output_path = (
            self.raw_data_path / "features" / f"{DataName.OFFLINE_FEATURES}.parquet"
        )
        self.key_collection = KeyDefinition().key_collection
        self.feature_query: List[str] = []

    def _query_feature(self):
        for feature_query in self.training_config.feature_queries:
            query = FeatureQuery(
                feature_list=feature_query.feature_names,
                key=[self.key_collection[i] for i in feature_query.key_names],
            )
            self.feature_query.append(query)

    def _get_offline_features(self):
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
        result_df = get_result_df(self.client, res_url=str(self.output_path))
        return result_df

    def run(self):
        self._query_feature()
        self._get_offline_features()
        # self._get_result_df()
