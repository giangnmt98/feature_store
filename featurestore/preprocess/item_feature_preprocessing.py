"""
Module: item_feature_preprocessing

This module is designed to preprocess item-related feature datasets, focusing on
content information. It implements transformation, normalization, and key construction
tasks for feature engineering.
"""
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.window import Window
from unidecode import unidecode

from configs import conf
from featurestore.base.feature_preprocessing import BaseDailyFeaturePreprocessing
from featurestore.base.utils.fileops import load_parquet_data
from featurestore.base.utils.utils import (
    norm_content_category,
    norm_content_category_by_pyspark,
)
from featurestore.constants import DataName


class ContentFeaturePreprocessing(BaseDailyFeaturePreprocessing):
    """
    Preprocesses content-related feature data.

    Attributes:
        process_lib (str): Library used for processing ('pandas' or 'pyspark').
        raw_data_path (str): Path to the raw data directory.
        save_filename (str): Filename for storing preprocessed content data.
        data_name_to_get_new_dates (str): Name of the dataset to retrieve new data.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.CONTENT_INFO,
        data_name_to_get_new_dates=DataName.CONTENT_INFO,
    ):
        super().__init__(
            process_lib, raw_data_path, save_filename, data_name_to_get_new_dates
        )

    def read_processed_data(self):
        df = self._load_raw_data(
            data_name=DataName.CONTENT_INFO,
        )
        content_type_df = load_parquet_data(
            file_paths=self.raw_data_dir / f"{DataName.CONTENT_TYPE}.parquet",
            process_lib=self.process_lib,
            spark=self.spark,
        )
        self.raw_data[DataName.CONTENT_INFO] = df
        self.raw_data[DataName.CONTENT_TYPE] = content_type_df

    def initialize_dataframe(self):
        df = self.create_item_key(self.raw_data[DataName.CONTENT_INFO])
        if self.process_lib in ["pandas"]:
            df = df.sort_values(by=["modifydate", "filename_date"], ascending=False)
            df = df.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
        else:
            df = (
                df.withColumn(
                    "create_date", F.date_format("create_date", "yyyy-MM-dd HH:mm:ss")
                )
                .withColumn(
                    "modifydate", F.date_format("modifydate", "yyyy-MM-dd HH:mm:ss")
                )
                .withColumn(
                    "publish_date", F.date_format("publish_date", "yyyy-MM-dd HH:mm:ss")
                )
            )
            df = (
                df.withColumn(
                    "row_number",
                    F.row_number().over(
                        Window.partitionBy("item_id").orderBy(
                            F.col("modifydate").desc(),
                            F.col("filename_date").desc(),
                        )
                    ),
                )
                .filter(F.col("row_number") == 1)
                .drop("row_number")
            )
        df = self._norm_some_content_cols(df)
        return df

    def _norm_some_content_cols(self, df):
        if self.process_lib in ["pandas"]:
            df.create_date = (
                df.create_date.dt.strftime("%Y%m")
                .fillna("200001")
                .astype(int)
                .astype(str)
            )
            if "content_category" in df.columns:
                df["content_category"] = df["content_category"].fillna("unknown")
            if "create_date" in df.columns:
                df["create_date"] = df["create_date"].fillna("unknown")
        else:
            df = df.withColumn(
                "create_date",
                F.date_format(F.col("create_date"), "yyyyMM")
                .cast(IntegerType())
                .cast(StringType()),
            )
            df = df.na.fill({"create_date": "200001"})

            if "content_category" in df.columns:
                df = df.na.fill(
                    {
                        "content_category": "unknown",
                    }
                )
            if "create_date" in df.columns:
                df = df.na.fill(
                    {
                        "create_date": "unknown",
                    }
                )
        return df

    def preprocess_content_country(self, df):
        """
        Preprocesses the content country information.

        Args:
            df (DataFrame): The input dataset containing content country information.

        Returns:
            DataFrame: The dataset with preprocessed and normalized information.
        """
        if self.process_lib in ["pandas"]:
            df["clean_content_country"] = (
                df["content_country"]
                .fillna("")
                .map(unidecode)
                .str.lower()
                .str.strip()
                .str.replace(" ", "_")
            )
            df["clean_content_country"] = df["clean_content_country"].replace(
                conf.clean_content_country_mapping
            )
            df.loc[
                df["item_id"].isin(["2#122737"]), "clean_content_country"
            ] = "viet_nam"
            df.loc[
                df["item_id"].isin(conf.nhat_ban_rulebase),
                "clean_content_country",
            ] = "nhat_ban"
            df.loc[
                df["item_id"].isin(conf.han_quoc_rulebase),
                "clean_content_country",
            ] = "han_quoc"
            df.loc[
                df["item_id"].isin(conf.tay_ban_nha_rulebase),
                "clean_content_country",
            ] = "tay_ban_nha"
            df.loc[
                df["item_id"].isin(conf.phap_rulebase),
                "clean_content_country",
            ] = "phap"
            df.loc[
                df["item_id"].isin(conf.trung_quoc_rulebase),
                "clean_content_country",
            ] = "trung_quoc"

            df.loc[df["item_id"].isin(["2#5040"]), "clean_content_country"] = "y"
            df.loc[df["item_id"].isin(["2#139070"]), "clean_content_country"] = "canada"
            df["clean_content_country"] = df["clean_content_country"].fillna("unknown")
        else:
            df = df.withColumn("clean_content_country", F.col("content_country"))
            df = df.na.fill({"clean_content_country": ""})
            df = df.withColumn(
                "clean_content_country",
                F.udf(unidecode, StringType())(F.col("clean_content_country")),
            )
            df = df.withColumn(
                "clean_content_country",
                F.regexp_replace(
                    F.trim(F.lower(F.col("clean_content_country"))), " ", "_"
                ),
            )
            df = df.replace(
                conf.clean_content_country_mapping, subset=["clean_content_country"]
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(
                    F.col("item_id").isin(["2#122737"]), F.lit("viet_nam")
                ).otherwise(F.col("clean_content_country")),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(
                    F.col("item_id").isin(conf.nhat_ban_rulebase), F.lit("nhat_ban")
                ).otherwise(F.col("clean_content_country")),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(
                    F.col("item_id").isin(conf.han_quoc_rulebase), F.lit("han_quoc")
                ).otherwise(F.col("clean_content_country")),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(
                    F.col("item_id").isin(conf.tay_ban_nha_rulebase),
                    F.lit("tay_ban_nha"),
                ).otherwise(F.col("clean_content_country")),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(
                    F.col("item_id").isin(conf.phap_rulebase), F.lit("phap")
                ).otherwise(F.col("clean_content_country")),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(
                    F.col("item_id").isin(conf.trung_quoc_rulebase), F.lit("trung_quoc")
                ).otherwise(F.col("clean_content_country")),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(F.col("item_id").isin(["2#5040"]), F.lit("y")).otherwise(
                    F.col("clean_content_country")
                ),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(F.col("item_id").isin(["2#139070"]), F.lit("canada")).otherwise(
                    F.col("clean_content_country")
                ),
            )
            df = df.na.fill(
                {
                    "clean_content_country": "unknown",
                }
            )
        return df

    def preprocess_content_category(self, df):
        """
        Preprocesses the content category information.

        Args:
            df (DataFrame): The input dataset containing content category information.

        Returns:
            DataFrame: The dataset with preprocessed and normalized information.
        """
        if self.process_lib in ["pandas"]:
            df["clean_content_category"] = df["content_category"].map(
                norm_content_category
            )
            df["clean_content_category"] = df["clean_content_category"].fillna(
                "unknown"
            )
        else:
            df = norm_content_category_by_pyspark(df)
            df = df.na.fill(
                {
                    "clean_content_category": "unknown",
                }
            )
        return df

    def preprocess_content_parent_type(self, df):
        """
        Preprocesses the content parent type information.

        Args:
            df (DataFrame): The input dataset containing content information.

        Returns:
            DataFrame: The dataset with preprocessed and normalized information.
        """
        if self.process_lib in ["pandas"]:
            content_type_df = self.raw_data[DataName.CONTENT_TYPE].rename(
                columns={"mapping": "content_parent_type"}
            )
            df = df.merge(content_type_df, on="content_type", how="left")
            df.loc[
                (df["content_single"] == 2) & (df["content_parent_type"] == "movie"),
                "content_parent_type",
            ] = "tv_series"
            df["content_parent_type"] = df["content_parent_type"].fillna("unknown")
        else:
            content_type_df = self.raw_data[DataName.CONTENT_TYPE].withColumnRenamed(
                "mapping", "content_parent_type"
            )
            df = df.join(content_type_df, on="content_type", how="left")
            df = df.withColumn(
                "content_parent_type",
                F.when(
                    (F.col("content_single") == 2)
                    & (F.col("content_parent_type") == "movie"),
                    F.lit("tv_series"),
                ).otherwise(F.col("content_parent_type")),
            )
            df = df.na.fill({"content_parent_type": "unknown"})
        return df

    def preprocess_is_content_type(self, df):
        """
        Preprocesses the content type information.

        Args:
            df (DataFrame): The input dataset containing content information.

        Returns:
            DataFrame: The dataset with preprocessed and normalized information.
        """
        if self.process_lib in ["pandas"]:
            df["is_channel_content"] = False
            df.loc[
                df["content_parent_type"].isin(["live", "tvod", "sport"]),
                "is_channel_content",
            ] = True

            df["is_vod_content"] = True
            df.loc[
                (df["content_parent_type"].isin(["movie", "tv_series"]))
                | (df["is_channel_content"]),
                "is_vod_content",
            ] = False

            df["is_movie_content"] = ~(df["is_vod_content"] | df["is_channel_content"])
        else:
            df = df.withColumn("is_channel_content", F.lit(False))
            df = df.withColumn(
                "is_channel_content",
                F.when(
                    F.col("content_parent_type").isin(["live", "tvod", "sport"]),
                    F.lit(True),
                ).otherwise(F.col("is_channel_content")),
            )
            df = df.withColumn("is_vod_content", F.lit(True))
            df = df.withColumn(
                "is_vod_content",
                F.when(
                    F.col("content_parent_type").isin(["movie", "tv_series"])
                    | F.col("is_channel_content"),
                    F.lit(False),
                ).otherwise(F.col("is_vod_content")),
            )
            df = df.withColumn(
                "is_movie_content",
                ~(F.col("is_vod_content") | F.col("is_channel_content")),
            )
        return df

    def preprocess_hashed_item_id(self, df):
        """
        Preprocesses the `hashed_item_id` field for content data based on the`item_id`

        Args:
            df (DataFrame): The input dataset containing content information.

        Returns:
            DataFrame: The dataset with additional columns for hashed item IDs.
        """
        spare_feature_info = conf.DhashSpareFeatureInfo()
        df = self.preprocess_hashed_id(
            df,
            output_feature_names=["hashed_item_id"],
            hash_dependency_info={"hashed_item_id": "item_id"},
            spare_feature_info=spare_feature_info,
        )
        df = self.preprocess_hashed_id(
            df,
            output_feature_names=["hashed_item_id_v2"],
            hash_dependency_info={"hashed_item_id_v2": "item_id"},
            spare_feature_info=spare_feature_info,
            version=2,
        )
        return df

    def preprocess_feature(self, df):
        df = self.preprocess_content_category(df)
        df = self.preprocess_content_country(df)
        df = self.preprocess_content_parent_type(df)
        df = self.preprocess_is_content_type(df)
        df = self.preprocess_hashed_item_id(df)
        return df
