"""
Module: source_definition

This module defines the `SourceDefinition` singleton class, which is responsible for
managing data sources used in the feature store pipeline. It centralizes the
configuration of batch and streaming data sources and includes metadata such as
file paths, timestamp information, and preprocessing steps. This ensures standardized
and reusable data source definitions across pipelines.
"""
from pathlib import Path

from feathr import HdfsSource

from configs.conf import TIMESTAMP_COLUMN, TIMESTAMP_FORMAT
from featurestore.base.utils.singleton import SingletonMeta
from featurestore.constants import DataName


class SourceDefinition(metaclass=SingletonMeta):
    """
    SourceDefinition is a singleton class for defining data sources
    used in the feature store pipeline.

    This class specifies batch and streaming data sources for features, describing
    their paths, metadata (e.g., timestamp columns), and preprocessing steps.

    Attributes:
        data_path (Path): Base directory containing the raw data files.
        user_source_path (str): Path to the user data file.
        content_info_source_path (str): Path to the content data file.
        online_user_source_path (str): Path to the online user feature data file.
        online_item_source_path (str): Path to the online item feature data file.
        ab_user_source_path (str): Path to the AB testing user data file.

        user_batch_source (HdfsSource): Batch source for user data.
        content_info_batch_source (HdfsSource): Batch source for content data.
        online_user_batch_source (HdfsSource): Batch source for online user features,
        includes timestamp and preprocessing.
        online_item_batch_source (HdfsSource): Batch source for online item features,
        includes timestamp and preprocessing.
        ab_user_batch_source (HdfsSource): Batch source for AB testing user data.
    """

    def __init__(
        self,
        data_path,
    ):
        self.data_path = Path(data_path)
        self.user_source_path = str(self.data_path / f"{DataName.USER_INFO}.parquet")
        self.content_info_source_path = str(
            self.data_path / f"{DataName.CONTENT_INFO}.parquet"
        )
        self.online_user_source_path = str(
            self.data_path / f"{DataName.ONLINE_USER_FEATURES}.parquet"
        )
        self.online_item_source_path = str(
            self.data_path / f"{DataName.ONLINE_ITEM_FEATURES}.parquet"
        )
        self.ab_user_source_path = str(
            self.data_path / f"{DataName.AB_TESTING_USER_INFO}.parquet"
        )

        self.user_batch_source = HdfsSource(
            name="userData",
            path=self.user_source_path,
        )
        self.content_info_batch_source = HdfsSource(
            name="contentInfoData",
            path=self.content_info_source_path,
        )
        self.online_user_batch_source = HdfsSource(
            name="onlineUserData",
            path=self.online_user_source_path,
            event_timestamp_column=TIMESTAMP_COLUMN,
            timestamp_format=TIMESTAMP_FORMAT,
            preprocessing=online_data_preprocessing,
        )
        self.online_item_batch_source = HdfsSource(
            name="onlineItemData",
            path=self.online_item_source_path,
            event_timestamp_column=TIMESTAMP_COLUMN,
            timestamp_format=TIMESTAMP_FORMAT,
            preprocessing=online_data_preprocessing,
        )
        self.ab_user_batch_source = HdfsSource(
            name="abUserData",
            path=self.ab_user_source_path,
        )


def online_data_preprocessing(df):
    """
    Filters the input DataFrame to include only rows within a specific date range.

    This function processes online data by applying a date-based filter on the
    `filename_date` column. The range is determined by a predefined interval
    from the current or configured end date.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame to preprocess.

    Returns:
        pyspark.sql.DataFrame: DataFrame containing rows within the specified date range

    Notes:
        In Feathr, all packages and variables must be imported inside
        the preprocessing method
    """
    # pylint: disable=C0415
    from datetime import datetime, timedelta

    from pyspark.sql.functions import col

    filename_date_col = "filename_date"
    filename_date_format = "%Y%m%d"
    online_interval = 90

    end_date = 20230516
    end_date = datetime.strptime(str(end_date), filename_date_format)
    # end_date = datetime.today()
    start_date = end_date - timedelta(days=online_interval)
    start_date = start_date.strftime(filename_date_format)
    end_date = end_date.strftime(filename_date_format)
    df = df.filter(col(filename_date_col) > start_date).filter(
        col(filename_date_col) <= end_date
    )
    return df
