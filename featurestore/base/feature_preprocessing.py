"""
Module: feature_preprocessing

This module defines the foundational classes and methods required for
feature preprocessing in both batch and online processing contexts.
It includes logic for loading raw data, applying feature transformations,
generating unique keys, and saving preprocessed data.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import LongType, StringType

from configs import conf
from featurestore.base.utils.fileops import load_parquet_data, save_parquet_data
from featurestore.base.utils.logger import logger
from featurestore.base.utils.spark import SparkOperations
from featurestore.base.utils.utils import get_full_date_list
from featurestore.constants import DataName
from featurestore.hashing_function import HashingClass


class BaseFeaturePreprocessing(ABC):
    """
    A base class for preprocessing features.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename="",
    ):
        """
        Initializes the feature preprocessing instance.

        Args:
            process_lib (str): The library to use for data processing.
            raw_data_path (str): Path to the directory containing raw data.
            save_filename (str): Name of the file to save preprocessed data.
        """
        self.process_lib = process_lib
        self.raw_data_dir = Path(raw_data_path)
        self.save_data_dir = self.raw_data_dir / "preprocessed_features"
        self.save_filename = save_filename
        self.raw_data: Dict[str, Any] = {}
        self.save_path = self.save_data_dir / f"{self.save_filename}.parquet"

        if self.process_lib == "pyspark":
            self.spark = SparkOperations().get_spark_session()
        else:
            self.spark = None

    @abstractmethod
    def read_processed_data(self):
        """
        Reads and retrieves processed data.

        This method is intended to read and return data that has already been processed.
        It serves as a placeholder for implementing logic to retrieve processed data
        from a specific source, such as a file, database, or any other storage.
        """
        raise NotImplementedError

    def create_user_key(self, df):
        """
        Creates and formats a unique user key for each record in the dataset.

        Args:
            df (DataFrame): The input dataset, which must contain `profile_id`
                and `username` columns.

        Returns:
            DataFrame: The dataset with an additional `user_id` column representing
                a unique user key.
        """
        if self.process_lib in ["pandas"]:
            df["user_id"] = df["profile_id"].copy()
            df.loc[df["user_id"] == -1, "user_id"] = 0
            df["user_id"] = df["user_id"].fillna(0)
            df["user_id"] = df["user_id"].astype(int).astype(str)
            df["username"] = df["username"].str.lower()
            df["user_id"] = df["user_id"] + "#" + df["username"]
            df = df[~df["user_id"].isnull()]
        else:
            df = df.withColumn("user_id", F.col("profile_id"))
            df = df.withColumn(
                "user_id",
                F.when(F.col("user_id").cast(StringType()) == "-1", "0").otherwise(
                    F.col("user_id")
                ),
            )
            df = df.na.fill({"user_id": "0"})
            df = df.withColumn("user_id", F.round(F.col("user_id")).cast(LongType()))
            df = df.withColumn("user_id", F.col("user_id").cast(StringType()))
            df = df.withColumn("username", F.lower(F.col("username")))
            df = df.withColumn(
                "user_id",
                F.concat(F.col("user_id"), F.lit("#"), F.col("username")),
            )
            df = df.filter(F.col("user_id").isNotNull())
        return df

    def create_item_key(self, df):
        """
        Creates and formats a unique item key for each record in the dataset.

        Args:
            df (DataFrame): The input dataset, which must contain`content_type`
                and `content_id` columns.

        Returns:
            DataFrame: The dataset with an additional `item_id` column
                representing a unique item key.
        """
        if self.process_lib in ["pandas"]:
            df["item_id"] = df["content_type"] + "#" + df["content_id"]
            df = df[~df["item_id"].isnull()]
        else:
            df = df.withColumn(
                "item_id",
                F.concat(F.col("content_type"), F.lit("#"), F.col("content_id")),
            )
            df = df.filter(F.col("item_id").isNotNull())
        return df

    @abstractmethod
    def initialize_dataframe(self):
        """
        Initializes a DataFrame.

        This method is designed to initialize a DataFrame before any processing.
        It returns the input DataFrame as is, serving as a placeholder for customization
        or additional setup.
        """
        raise NotImplementedError

    def preprocess_feature(self, df):
        """
        Preprocesses a feature DataFrame.

        This method is intended to preprocess features in a provided DataFrame.
        It currently returns the input DataFrame without any modifications,
        acting as a placeholder for future preprocessing logic.
        """
        return df

    def preprocess_hashed_id(
        self,
        df,
        output_feature_names,
        hash_dependency_info,
        spare_feature_info=conf.SpareFeatureInfo(),
        version=1,
    ):
        """
        Preprocesses and applies hashing functions to specified features.

        Args:
            df (DataFrame): The input dataset to be processed.
            output_feature_names (list): List of feature names to apply hashing on.
            hash_dependency_info (dict): Dependency mapping for each output feature.
            spare_feature_info (SpareFeatureInfo, optional): Config for hashed features
            version (int, optional): Version of the hashing function. Defaults to 1.

        Returns:
            DataFrame: The dataset with hashed features added.
        """
        if self.process_lib in ["pandas"]:
            for output_feature in output_feature_names:
                dependency_col, hash_bucket_size = self._get_hash_dependency_info(
                    output_feature, hash_dependency_info, spare_feature_info
                )
                df = HashingClass(self.raw_data_dir).hashing_func(
                    df,
                    output_feature,
                    dependency_col,
                    hash_bucket_size,
                    "pandas",
                    version,
                )
        else:
            for output_feature in output_feature_names:
                dependency_col, hash_bucket_size = self._get_hash_dependency_info(
                    output_feature, hash_dependency_info, spare_feature_info
                )
                df = HashingClass(self.raw_data_dir).hashing_func(
                    df,
                    output_feature,
                    dependency_col,
                    hash_bucket_size,
                    "pyspark",
                    version,
                )
        return df

    def _get_hash_dependency_info(
        self,
        feature_name,
        hash_dependency_info,
        spare_feature_info=conf.SpareFeatureInfo(),
    ):
        """
        Retrieves hash dependency information for a specific feature.

        Args:
            feature_name (str): The name of the feature to retrieve dependency info for.
            hash_dependency_info (dict): Dictionary mapping features to their
                dependencies.
            spare_feature_info (SpareFeatureInfo, optional): Configuration for
                hashed features

        Returns:
            tuple: A tuple containing the dependency column and hash bucket size.
        """
        # validate hash_dependency_info
        dependency_col = hash_dependency_info.get(feature_name, None)
        hash_bucket_size = spare_feature_info.hashed_features.get(feature_name, None)
        assert (
            dependency_col is not None
        ), f"Cannot found dependency feature of feature {feature_name}"
        assert (
            hash_bucket_size is not None
        ), f"Cannot found hash bucket size of feature {feature_name}"
        return dependency_col, hash_bucket_size

    def save_preprocessed_data(self, df):
        """
        Saves the preprocessed dataset to disk.

        Args:
            df (DataFrame): The preprocessed dataset to save.
        """
        save_parquet_data(
            df,
            save_path=self.save_path,
            process_lib=self.process_lib,
        )

    def run(self):
        """
        Executes the full preprocessing pipeline.
        """
        logger.info(f"Start preprocess features to {self.save_path}")
        self.read_processed_data()
        df = self.initialize_dataframe()
        df = self.preprocess_feature(df)
        self.save_preprocessed_data(df)


class BaseDailyFeaturePreprocessing(BaseFeaturePreprocessing):
    """
    Base class for daily feature preprocessing tasks.

    This class extends the `BaseFeaturePreprocessing` class and provides additional
    methods for daily feature extraction. It is designed to handle datasets
    partitioned by dates and focuses on processing newly available data while
    avoiding reprocessing previously processed dates.

    Attributes:
        process_lib (str): The library used for data processing ('pandas' or 'pyspark').
            Defaults to "pandas".
        raw_data_path (str): The path to the directory containing raw data. Defaults
            to "data/processed/".
        save_filename (str): The name of the file to save preprocessed data.
        data_name_to_get_new_dates (str): The name of the dataset used for identifying
            dates to process.
        filename_date_col (str): The column representing dates in the dataset, typically
            used for filtering data.
        dates_to_extract (List[int] | Set[int] | None): A list or set of new dates to
            process, or None if there is no new data.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename="",
        data_name_to_get_new_dates="",
    ):
        super().__init__(process_lib, raw_data_path, save_filename)
        self.filename_date_col = conf.FILENAME_DATE_COL
        self.data_name_to_get_new_dates = data_name_to_get_new_dates
        self.dates_to_extract = self._get_new_dates()

    def _get_new_dates(self):
        """
        Computes the new dates to process by comparing existing and raw data.

        Returns:
            List[int] | Set[int] | None: A sorted list of new dates to process,
                or None if no raw data is found.
        """
        raw_data_path = self.raw_data_dir / f"{self.data_name_to_get_new_dates}.parquet"
        save_data_path = self.save_data_dir / f"{self.save_filename}.parquet"
        if not raw_data_path.exists():
            return None
        raw_dates = {
            int(file_path.name.split("=")[-1])
            for file_path in raw_data_path.glob(f"{self.filename_date_col}*")
        }
        saved_dates = {
            int(file_path.name.split("=")[-1])
            for file_path in save_data_path.glob(f"{self.filename_date_col}*")
        }
        new_dates: Union[List[int], Set[int]] = raw_dates - saved_dates
        new_dates = sorted(list(new_dates))
        return new_dates

    def _load_raw_data(
        self,
        data_name: str,
        with_columns: Optional[List[str]] = None,
        dates_to_extract: Optional[List] = None,
        filters: Optional[List] = None,
        schema: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Loads raw data for processing using either Pandas or PySpark.

        Args:
            data_name (str): Name of the raw data file (without extension) to load.
            with_columns (list, optional): List of specific columns to load.
            filters (list, optional): Filtering conditions for the data.
            schema (any, optional): Schema to apply when loading data with PySpark.

        Returns:
            pd.DataFrame: The loaded raw dataset ready for processing.
        """
        assert (
            dates_to_extract is None or len(dates_to_extract) > 0
        ), "dates_to_extract must be not empty"
        data_path = self.raw_data_dir / f"{data_name}.parquet"
        date_filters = (
            [(self.filename_date_col, "in", dates_to_extract)]
            if dates_to_extract
            else None
        )
        if filters:
            filters = (filters + date_filters) if date_filters else filters
        else:
            filters = date_filters
        if self.process_lib == "pandas":
            pandas_filters = filters
            df = load_parquet_data(
                file_paths=data_path,
                with_columns=with_columns,
                process_lib=self.process_lib,
                filters=pandas_filters,
            )
            if len(df) == 0:
                raise ValueError(
                    f"No data found in " f"{data_path} with filters: {pandas_filters}"
                )
        else:
            pyspark_filters = (
                [
                    (
                        item[0],
                        item[1],
                        [str(it) for it in item[2]]
                        if (isinstance(item[2], list) and (item[0] == "filename_date"))
                        else item[2],
                    )
                    for item in filters
                ]
                if filters
                else None
            )
            df = load_parquet_data(
                file_paths=data_path,
                with_columns=with_columns,
                process_lib=self.process_lib,
                filters=pyspark_filters,
                spark=self.spark,
                schema=schema,
            )
            if df.rdd.isEmpty():
                raise ValueError(
                    f"No data found in " f"{data_path} with filters: {pyspark_filters}"
                )
        return df

    def save_preprocessed_data(self, df):
        """
        Saves the preprocessed dataset to disk.

        Args:
            df (DataFrame): The preprocessed dataset to save.
        """
        save_parquet_data(
            df,
            save_path=self.save_path,
            process_lib=self.process_lib,
            partition_cols=self.filename_date_col,
            overwrite=False,
        )

    def run(self):
        """
        Executes the full preprocessing pipeline.
        """
        if self.dates_to_extract is None:
            logger.warning(f"Can not found {self.save_path}. Loading all data")
        elif len(self.dates_to_extract) == 0:
            logger.info("No new data found. Skip extract features")
            return
        else:
            logger.info(f"Loading raw data from date: {self.dates_to_extract}")
        super().run()


class BaseOnlineFeaturePreprocessing(BaseDailyFeaturePreprocessing):
    """
    Base class for online feature preprocessing tasks.

    Online features are dynamically extracted for real-time updates and modeling.

    Attributes:
        process_lib (str): Library used for processing ('pandas' or 'pyspark').
        raw_data_path (str): Path to the raw data directory.
        save_filename (str): Filename for saving preprocessed online features.
        data_name_to_get_new_dates (str): Dataset name to get updated daily online data.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename="",
        data_name_to_get_new_dates=DataName.MOVIE_HISTORY,
    ):
        super().__init__(
            process_lib,
            raw_data_path,
            save_filename,
            data_name_to_get_new_dates,
        )

    def _get_full_date_to_extract(self):
        full_date_list = []
        for d in self.dates_to_extract:
            d = pd.to_datetime(d, format=conf.FILENAME_DATE_FORMAT)
            date_list = get_full_date_list(
                for_date=d, num_days_before=conf.ROLLING_PERIOD_FOR_POPULARITY_ITEM
            )
            full_date_list += date_list

        full_dates_to_extract = list(set(full_date_list))
        return full_dates_to_extract

    def read_processed_data(self):
        full_dates_to_extract = self._get_full_date_to_extract()
        movie_df = self._load_raw_data(
            data_name=DataName.MOVIE_HISTORY,
            with_columns=conf.SELECTED_HISTORY_COLUMNS,
            dates_to_extract=full_dates_to_extract,
        )
        vod_df = self._load_raw_data(
            data_name=DataName.VOD_HISTORY,
            with_columns=conf.SELECTED_HISTORY_COLUMNS,
            dates_to_extract=full_dates_to_extract,
        )
        self.raw_data[DataName.MOVIE_HISTORY] = movie_df
        self.raw_data[DataName.VOD_HISTORY] = vod_df

    def initialize_dataframe(self):
        if self.process_lib == "pandas":
            df = pd.concat(
                [
                    self.raw_data[DataName.MOVIE_HISTORY],
                    self.raw_data[DataName.VOD_HISTORY],
                ]
            )
        else:
            df = self.raw_data[DataName.MOVIE_HISTORY].union(
                self.raw_data[DataName.VOD_HISTORY]
            )
        df = self.create_user_key(df)
        df = self.create_item_key(df)
        return df
