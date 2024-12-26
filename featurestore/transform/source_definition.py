from pathlib import Path

from feathr import HdfsSource

from configs.conf import TIMESTAMP_COLUMN, TIMESTAMP_FORMAT
from featurestore.base.utils.singleton import SingletonMeta
from featurestore.constants import DataName


class SourceDefinition(metaclass=SingletonMeta):
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
    from datetime import datetime, timedelta

    from pyspark.sql.functions import col

    FILENAME_DATE_COL = "filename_date"
    FILENAME_DATE_FORMAT = "%Y%m%d"
    ONLINE_INTERVAL = 90

    end_date = 20230516
    end_date = datetime.strptime(str(end_date), FILENAME_DATE_FORMAT)
    # end_date = datetime.today()
    start_date = end_date - timedelta(days=ONLINE_INTERVAL)
    start_date = start_date.strftime(FILENAME_DATE_FORMAT)
    end_date = end_date.strftime(FILENAME_DATE_FORMAT)
    df = df.filter(col(FILENAME_DATE_COL) > start_date).filter(
        col(FILENAME_DATE_COL) <= end_date
    )
    return df
