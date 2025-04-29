from featurestore.base.feature_preprocessing import BaseFeaturePreprocessing
from featurestore.base.utils.fileops import load_parquet_data
from featurestore.constants import DataName


class ABUserFeaturePreprocessing(BaseFeaturePreprocessing):
    """
    Preprocesses user data for A/B testing.

    Attributes:
        process_lib (str): Library used for processing ('pandas' or 'pyspark').
        raw_data_path (str): Path to the raw data directory.
        save_filename (str): Filename for storing A/B testing user data.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.AB_TESTING_USER_INFO,
        spark_config=None,
    ):
        super().__init__(process_lib, raw_data_path, save_filename, spark_config)

    def read_processed_data(self):
        data_path = self.raw_data_dir / f"{DataName.AB_TESTING_USER_INFO}.parquet"
        df = load_parquet_data(
            file_paths=data_path,
            process_lib=self.process_lib,
            spark=self.spark,
        )
        self.raw_data[DataName.AB_TESTING_USER_INFO] = df

    def initialize_dataframe(self):
        df = self.raw_data[DataName.AB_TESTING_USER_INFO].select(
            ["profile_id", "group_id"]
        )
        return df
