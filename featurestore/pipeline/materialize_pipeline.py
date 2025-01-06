from datetime import datetime, timedelta

import pandas as pd
from feathr import BackfillTime, HdfsSink, MaterializationSettings, RedisSink

from configs.conf import FILENAME_DATE_FORMAT
from featurestore.base.schemas.pipeline_config import MaterializePipelineConfig
from featurestore.base.utils.config import parse_materialize_config
from featurestore.base.utils.utils import get_omitted_date, return_or_load


class MaterializePipeline:
    def __init__(
        self,
        config_path: str,
        feathr_client,
        raw_data_path: str,
        materialize_for_eval: bool = False,
        num_date_eval=3,
    ):
        self.materialize_config = return_or_load(
            config_path, MaterializePipelineConfig, parse_materialize_config
        )
        self.client = feathr_client
        self.save_offline_materialized_path = (
            raw_data_path + "/" + self.materialize_config.save_dir_path
        )
        self.materialize_for_eval = materialize_for_eval
        infer_date = self.materialize_config.infer_date
        self.num_date_eval = num_date_eval
        if infer_date == "today":
            self.infer_date = datetime.today()
        else:
            self.infer_date = pd.to_datetime(infer_date, format=FILENAME_DATE_FORMAT)

        if self.materialize_for_eval:
            self.execution_configurations = (
                self.materialize_config.offline_spark_execution_configs
            )
        else:
            self.execution_configurations = (
                self.materialize_config.online_spark_execution_configs
            )

    def _get_backfill_config(self, table_name_list):
        backfill_time = None
        if self.materialize_for_eval:
            omit_date_list = get_omitted_date(
                for_date=self.infer_date,
                folder_path=self.save_offline_materialized_path
                + f"/{table_name_list[0]}",
                num_days_before=self.num_date_eval,
            )
            if len(omit_date_list) > 0:
                backfill_time = BackfillTime(
                    start=pd.to_datetime(
                        min(omit_date_list), format=FILENAME_DATE_FORMAT
                    ),
                    end=self.infer_date,
                    step=timedelta(days=1),
                )
        return backfill_time

    def _get_sink_list(self, table_name_list):
        sink_list = []
        if self.materialize_for_eval:
            dir_path = self.save_offline_materialized_path
            for table_name in table_name_list:
                save_path = dir_path + f"/{table_name}"
                hdfs_sink = HdfsSink(
                    output_path=save_path,
                )
                sink_list.append(hdfs_sink)
        else:
            for table_name in table_name_list:
                redis_sink = RedisSink(table_name=table_name)
                sink_list.append(redis_sink)
        return sink_list

    def _materialize_features(self, table_name_list, setting_name, feature_names):
        backfill_time = self._get_backfill_config(table_name_list)
        sink_list = self._get_sink_list(table_name_list)
        settings = MaterializationSettings(
            name=setting_name,
            sinks=sink_list,
            feature_names=feature_names,
            backfill_time=backfill_time,
        )
        self.client.materialize_features(
            settings=settings,
            allow_materialize_non_agg_feature=True,
            execution_configurations=self.execution_configurations,
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
