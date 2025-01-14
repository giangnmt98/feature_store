"""
Module: materialize_pipeline

This module provides the `MaterializePipeline` class, which is responsible for the
materialization of features from a feature store into online or offline storage.
The pipeline leverages configurations and feature store clients to streamline
materialization processes, supporting both evaluation and production modes.
"""
import shutil
from datetime import datetime, timedelta
from glob import glob

import pandas as pd
from distutils.dir_util import copy_tree
from feathr import BackfillTime, HdfsSink, MaterializationSettings, RedisSink

from configs.conf import FILENAME_DATE_FORMAT
from featurestore.base.schemas.pipeline_config import MaterializePipelineConfig
from featurestore.base.utils.config import parse_materialize_config
from featurestore.base.utils.utils import get_omitted_date, return_or_load


class MaterializePipeline:
    """
    MaterializePipeline is responsible for materializing features to online or offline
    storage based on the provided configurations.

    Attributes:
        materialize_config (MaterializePipelineConfig): Parsed configuration object
        client: Feathr client for interacting with the feature store.
        save_offline_materialized_path (str): Path to store materialized features in
            offline mode.
        materialize_for_eval (bool): If True, runs the pipeline in evaluation mode
            and materializes features to offline sinks.
        infer_date (datetime): Date used for feature materialization, parsed from the
            configuration.
        execution_configurations (dict): Spark execution configurations for online
            or offline materialization.
    """

    def __init__(
        self,
        config_path: str,
        feathr_client,
        raw_data_path: str,
        infer_date: str,
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
        self.temp_save_path = self.save_offline_materialized_path + "/temp"
        self.materialize_for_eval = materialize_for_eval
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
        """
        Configures the backfill time range for processing features in evaluation mode.

        This method identifies the necessary backfill dates by checking omitted dates
        in existing offline materialized data. The backfill configuration defines
        a time range and step for materializing features.

        Args:
            table_name_list (list): List of table names to which backfill
            configuration is applied.

        Returns:
            BackfillTime or None: A BackfillTime object specifying the start,
            end, and step for backfill processing if backfill is required.
            Returns None if no backfill is needed.
        """
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
        """
        Prepares a list of output sinks for materialized features based on the
        pipeline's mode.

        In evaluation mode (offline), features are saved to an HDFS-compatible path.
        In production mode (online), features are stored in Redis.

        Args:
            table_name_list (list): List of table names to use when setting up
            output sinks.

        Returns:
            list: A list of sink objects configured for the given table names.
        """
        sink_list = []
        if self.materialize_for_eval:
            for table_name in table_name_list:
                save_path = self.temp_save_path + f"/{table_name}"
                hdfs_sink = HdfsSink(
                    output_path=save_path,
                    # store_name="df",
                )
                sink_list.append(hdfs_sink)
        else:
            for table_name in table_name_list:
                redis_sink = RedisSink(table_name=table_name)
                sink_list.append(redis_sink)
        return sink_list

    def _materialize_features(self, table_name_list, setting_name, feature_names):
        """
        Materializes features from the feature store for the given feature tables.

        Args:
            table_name_list (list): List of table names for which features need to be
            materialized.
            setting_name (str): A unique name to apply to the materialization setting.
            feature_names (list): List of feature names to materialize.
        """
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

    def materialize_all_table(self):
        """
        Iterates through all feature tables defined in the configuration
        and materializes their features.
        """
        for table in self.materialize_config.feature_tables:
            self._materialize_features(
                table_name_list=table.feature_table_names,
                setting_name=table.setting_name,
                feature_names=table.feature_names,
            )
        if self.materialize_for_eval:
            self._postprocesss_save_folder()

    def _postprocesss_save_folder(self):
        all_table_name = []
        for table in self.materialize_config.feature_tables:
            all_table_name.extend(table.feature_table_names)

        for table_name in all_table_name:
            subfolder_list = glob(
                self.temp_save_path + f"/{table_name}/df*/daily/*/*/*"
            )
            existed_subfolder_list = glob(
                self.save_offline_materialized_path + f"/{table_name}/*"
            )
            existed_date_list = [i.split("/")[-1] for i in existed_subfolder_list]
            # copy subdirectory
            for folder_path in subfolder_list:
                new_date_path = folder_path.split("daily/")[-1].replace("/", "")
                if new_date_path not in existed_date_list:
                    new_folder_path = (
                        self.save_offline_materialized_path
                        + f"/{table_name}/{new_date_path}"
                    )
                    copy_tree(folder_path, new_folder_path)
        # delete temp folder
        shutil.rmtree(self.temp_save_path)

    def run(self):
        """
        Executes the entire materialization pipeline.
        """
        self.materialize_all_table()
