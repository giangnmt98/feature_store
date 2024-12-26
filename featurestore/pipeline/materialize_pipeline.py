from feathr import MaterializationSettings, RedisSink

from featurestore.base.schemas.pipeline_config import MaterializePipelineConfig
from featurestore.base.utils.config import parse_materialize_config
from featurestore.base.utils.utils import return_or_load


class MaterializePipeline:
    def __init__(
        self,
        config_path: str,
        feathr_client,
    ):
        self.materialize_config = return_or_load(
            config_path, MaterializePipelineConfig, parse_materialize_config
        )
        self.client = feathr_client

    def _materialize_features(self, table_name_list, setting_name, feature_names):
        redis_sink_list = []
        for table_name in table_name_list:
            redis_sink = RedisSink(table_name=table_name)
            redis_sink_list.append(redis_sink)

        settings = MaterializationSettings(
            name=setting_name,
            sinks=redis_sink_list,
            feature_names=feature_names,
        )
        self.client.materialize_features(
            settings=settings,
            allow_materialize_non_agg_feature=True,
            execution_configurations=self.materialize_config.spark_execution_config,
        )
        self.client.wait_job_to_finish(timeout_sec=300)

    def _materialize_all_table(self):
        for table in self.materialize_config.feature_tables:
            self._materialize_features(
                table_name_list=table.feature_table_names,
                setting_name=table.setting_name,
                feature_names=table.feature_names,
            )

    def run(self):
        self._materialize_all_table()
