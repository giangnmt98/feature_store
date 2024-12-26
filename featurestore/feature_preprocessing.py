from typing import Union

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StringType, StructType
from pyspark.sql.window import Window
from tqdm import tqdm
from unidecode import unidecode

from configs import conf
from featurestore.base.feature_preprocessing import BaseFeaturePreprocessing
from featurestore.base.utils.fileops import load_parquet_data, save_parquet_data
from featurestore.base.utils.logger import logger
from featurestore.base.utils.utils import split_batches
from featurestore.constants import DataName
from featurestore.daily_data_utils import get_date_before


class AccountFeaturePreprocessing(BaseFeaturePreprocessing):
    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
    ):
        super().__init__(process_lib, raw_data_path)

    def read_processed_data(self):
        data_path = self.raw_data_dir / f"{DataName.ACCOUNT_MYTV_INFO}.parquet"
        df = load_parquet_data(
            file_paths=data_path,
            process_lib=self.process_lib,
            spark=self.spark,
        )
        return df

    def initialize_dataframe(self, df):
        # create feature values for new users
        if self.process_lib == "pandas":
            new_values = {
                "username": ["empty"],
                "province": ["HCM"],
                "package_code": ["MYTV021"],
                "sex": ["1"],
                "age": [0],
                "platform": ["b2c-android"],
                "profile_id": [0],
            }
            new_df = pd.DataFrame(data=new_values)
            df = pd.concat([new_df, df])
        else:
            columns = [
                "username",
                "province",
                "package_code",
                "sex",
                "age",
                "platform",
                "profile_id",
            ]
            vals = [("empty", "HCM", "MYTV021", "1", 0, "b2c-android", 0)]
            new_df = self.spark.createDataFrame(vals, columns)
            df = new_df.union(df.select(columns))
        return df

    def preprocess_feature(self, user_info):
        valid_package_code = {
            "MYTV006": "Goi nang cao",
            "MYTV021": "Goi nang cao moi",
            "MYTV008": "Goi Vip",
            "MYTV010": "B2B Chuan",
            "MYTV014": "B2B Gold",
            "MYTV015": "B2B Premium",
            "MYTV013": "B2B Standard",
            "MYTV016": "B2B Diamond",
            "MYTV012": "B2B Vip",
            "FLX001": "mytv flexi",
            "FLX003": "Smart Flexi",
            "VASC006": "Mytv silver hd",
            "VASC005": "Mytv silver",
            "VASC007": "Mytv Gold",
            "VASC011": "Goi chuan",  # Galaxy thu 7
            "VASC000": "mytv basic",  # Galaxy thu 7
            "MYTV020": "Goi chuan moi",  # Galaxy thu 7
            "MYTV019": "Goi Co ban moi",  # Galaxy thu 7
        }
        if self.process_lib in ["pandas"]:
            user_info.loc[(user_info.age <= 5) | (user_info.age >= 95), "age"] = np.nan

            user_info["age_group"] = np.nan
            user_info.loc[user_info.age < 15, "age_group"] = "child"
            user_info.loc[
                (user_info.age >= 15) & (user_info.age < 22), "age_group"
            ] = "student"
            user_info.loc[
                (user_info.age >= 22) & (user_info.age < 30), "age_group"
            ] = "play"
            user_info.loc[
                (user_info.age >= 30) & (user_info.age < 40), "age_group"
            ] = "married"
            user_info.loc[
                (user_info.age >= 40) & (user_info.age < 65), "age_group"
            ] = "senior"
            user_info.loc[(user_info.age >= 65), "age_group"] = "older"
            user_info.loc[
                ~user_info.package_code.isin(valid_package_code.keys()), "package_code"
            ] = np.nan
        else:
            user_info = user_info.withColumn(
                "age",
                F.when(
                    (F.col("age") >= 95) | (F.col("age") <= 5), F.lit(None)
                ).otherwise(F.col("age")),
            )
            user_info = user_info.withColumn("age_group", F.lit(None))
            user_info = user_info.withColumn(
                "age_group",
                F.when(F.col("age") < 15, F.lit("child")).otherwise(F.col("age_group")),
            )
            user_info = user_info.withColumn(
                "age_group",
                F.when(
                    (F.col("age") >= 15) & (F.col("age") < 22), F.lit("student")
                ).otherwise(F.col("age_group")),
            )
            user_info = user_info.withColumn(
                "age_group",
                F.when(
                    (F.col("age") >= 22) & (F.col("age") < 30), F.lit("play")
                ).otherwise(F.col("age_group")),
            )
            user_info = user_info.withColumn(
                "age_group",
                F.when(
                    (F.col("age") >= 30) & (F.col("age") < 40), F.lit("married")
                ).otherwise(F.col("age_group")),
            )
            user_info = user_info.withColumn(
                "age_group",
                F.when(
                    (F.col("age") >= 40) & (F.col("age") < 65), F.lit("senior")
                ).otherwise(F.col("age_group")),
            )
            user_info = user_info.withColumn(
                "age_group",
                F.when((F.col("age") >= 65), F.lit("older")).otherwise(
                    F.col("age_group")
                ),
            )
            user_info = user_info.withColumn(
                "package_code",
                F.when(
                    F.col("package_code").isin(list(valid_package_code.keys())),
                    F.col("package_code"),
                ).otherwise(F.lit("None")),
            )
        return user_info

    def run(self):
        df = self.read_processed_data()
        df = self.initialize_dataframe(df)
        df = self.preprocess_feature(df)
        return df


class ProfileFeaturePreprocessing(BaseFeaturePreprocessing):
    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
    ):
        super().__init__(process_lib, raw_data_path)

    def read_processed_data(self):
        data_path = self.raw_data_dir / f"{DataName.PROFILE_MYTV_INFO}.parquet"
        df = load_parquet_data(
            file_paths=data_path,
            process_lib=self.process_lib,
            spark=self.spark,
        )
        return df

    def initialize_dataframe(self, df):
        df = self._create_data_for_new_user(df)
        df = self.create_user_key(df)
        if self.process_lib in ["pandas"]:
            df = df.drop_duplicates()
        else:
            df = df.dropDuplicates()
        return df

    def _create_data_for_new_user(self, df):
        if self.process_lib == "pandas":
            new_values = {
                "profile_id": [0],
                "username": ["empty"],
            }
            new_df = pd.DataFrame(data=new_values)
            df = pd.concat([new_df, df])
        else:
            columns = ["profile_id", "username"]
            vals = [(0, "empty")]
            new_df = self.spark.createDataFrame(vals, columns)
            df = new_df.union(df.select(columns))
        return df

    def preprocess_feature(self, df):
        spare_feature_info = conf.DhashSpareFeatureInfo()
        output_feature_names = ["hashed_user_id"]
        hash_dependency_info = {"hashed_user_id": "user_id"}
        df = self.preprocess_hashed_id(
            df,
            output_feature_names,
            hash_dependency_info,
            spare_feature_info,
        )
        output_feature_names = ["hashed_user_id_v2"]
        hash_dependency_info = {"hashed_user_id_v2": "user_id"}
        df = self.preprocess_hashed_id(
            df,
            output_feature_names,
            hash_dependency_info,
            spare_feature_info,
            version=2,
        )
        return df

    def run(self):
        df = self.read_processed_data()
        df = self.initialize_dataframe(df)
        df = self.preprocess_feature(df)
        return df


class UserFeaturePreprocessing(BaseFeaturePreprocessing):
    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.USER_INFO,
    ):
        super().__init__(process_lib, raw_data_path, save_filename)

    def read_processed_data(self):
        account_df = AccountFeaturePreprocessing(
            self.process_lib, self.raw_data_dir
        ).run()
        profile_df = ProfileFeaturePreprocessing(
            self.process_lib, self.raw_data_dir
        ).run()
        if self.process_lib == "pandas":
            account_df = account_df.drop(columns=["profile_id"])
            df = profile_df.merge(account_df, on="username", how="left")
        else:
            account_df = account_df.drop(F.col("profile_id"))
            df = profile_df.join(account_df, on="username", how="left")
        return df


class ABUserFeaturePreprocessing(BaseFeaturePreprocessing):
    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.AB_TESTING_USER_INFO,
    ):
        super().__init__(process_lib, raw_data_path, save_filename)

    def read_processed_data(self):
        data_path = self.raw_data_dir / f"{DataName.AB_TESTING_USER_INFO}.parquet"
        df = load_parquet_data(
            file_paths=data_path,
            process_lib=self.process_lib,
            spark=self.spark,
        )
        return df

    def initialize_dataframe(self, df):
        df = self.create_user_key(df)
        return df


class ContentFeaturePreprocessing(BaseFeaturePreprocessing):
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
        self.content_type_df = load_parquet_data(
            file_paths=self.raw_data_dir / f"{DataName.CONTENT_TYPE}.parquet",
            process_lib=self.process_lib,
            spark=self.spark,
        )

    def read_processed_data(self):
        df = self._load_raw_data(
            data_name=DataName.CONTENT_INFO,
        )
        return df

    def initialize_dataframe(self, df):
        df = self.create_item_key(df)
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
        clean_content_country_mapping = {
            # viet_nam
            "vietnam": "viet_nam",
            "việt_nam": "viet_nam",
            "1": "viet_nam",
            "vn": "viet_nam",
            "mỹ": "my",
            "america": "my",
            "6": "my",
            "6,7": "my",
            "6,8": "my",
            "6,7,11": "my",
            "6,7,20": "my",
            "6,11": "my",
            "6,7,8": "my",
            "11,6": "my",
            "europe_american": "my",
            # trung_quoc
            "china_mainland": "trung_quoc",
            "3": "trung_quoc",
            "china": "trung_quoc",
            # hong_kong
            "trung_quoc_hong_kong": "hong_kong",
            "hongkong_china": "hong_kong",
            "12": "hong_kong",
            # dai_loan
            "trung_quoc_dai_loan": "dai_loan",
            "dai_loan_trung_quoc": "dai_loan",
            "taiwan_china": "dai_loan",
            "13": "dai_loan",
            "dai_loan_(trung_quoc)": "dai_loan",
            # nhat_ban
            "japan": "nhat_ban",
            "14": "nhat_ban",
            "nhat_-_han": "nhat_ban",
            # han_quoc
            "korea": "han_quoc",
            "35": "han_quoc",
            "15": "han_quoc",
            # anh
            "british": "anh",
            "7": "anh",
            # phap
            "france": "phap",
            # duc
            "8": "duc",
            "11": "duc",
            "germany": "duc",
            # nga
            "russia": "nga",
            "10": "nga",
            # y
            "italia": "y",
            "italy": "y",
            # an_do
            "india": "an_do",
            "16": "an_do",
            # canada
            "23": "canada",
            # dan_mach
            "19": "dan_mach",
            # philippines
            "philippin": "philippines",
            # most contents are in america
            "17": "my",
            # empty country
            "": "empty",
            # israel
            "isarel": "israel",
            # other
            "khac": "other",
            "nuoc_khac": "other",
        }
        nhat_ban_rulebase = [
            "6#52740",
            "6#33346",
            "6#25451",
            "6#25445",
            "6#25418",
            "3#25418",
            "3#52740",
            "3#33346",
            "6#59571",
            "2#125213",
            "6#52541",
            "3#52541",
            "3#59571",
        ]
        han_quoc_rulebase = [
            "6#31759",
            "6#31745",
            "6#27593",
            "6#31781",
            "6#31771",
            "2#138664",
        ]
        tay_ban_nha_rulebase = [
            "6#25444",
            "3#25444",
            "20#139384",
            "20#130479",
            "20#136589",
            "20#130297",
            "2#6417",
        ]
        phap_rulebase = ["2#138544", "2#138543", "2#138002", "2#138161"]
        trung_quoc_rulebase = [
            "2#139154",
            "2#139153",
            "2#139155",
            "2#139156",
            "2#123813",
            "2#123808",
        ]
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
                clean_content_country_mapping
            )
            df.loc[
                df["item_id"].isin(["2#122737"]), "clean_content_country"
            ] = "viet_nam"
            df.loc[
                df["item_id"].isin(nhat_ban_rulebase),
                "clean_content_country",
            ] = "nhat_ban"
            df.loc[
                df["item_id"].isin(han_quoc_rulebase),
                "clean_content_country",
            ] = "han_quoc"
            df.loc[
                df["item_id"].isin(tay_ban_nha_rulebase),
                "clean_content_country",
            ] = "tay_ban_nha"
            df.loc[
                df["item_id"].isin(phap_rulebase),
                "clean_content_country",
            ] = "phap"
            df.loc[
                df["item_id"].isin(trung_quoc_rulebase),
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
                clean_content_country_mapping, subset=["clean_content_country"]
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
                    F.col("item_id").isin(nhat_ban_rulebase), F.lit("nhat_ban")
                ).otherwise(F.col("clean_content_country")),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(
                    F.col("item_id").isin(han_quoc_rulebase), F.lit("han_quoc")
                ).otherwise(F.col("clean_content_country")),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(
                    F.col("item_id").isin(tay_ban_nha_rulebase), F.lit("tay_ban_nha")
                ).otherwise(F.col("clean_content_country")),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(F.col("item_id").isin(phap_rulebase), F.lit("phap")).otherwise(
                    F.col("clean_content_country")
                ),
            )
            df = df.withColumn(
                "clean_content_country",
                F.when(
                    F.col("item_id").isin(trung_quoc_rulebase), F.lit("trung_quoc")
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
        if self.process_lib in ["pandas"]:
            df["clean_content_category"] = df["content_category"].map(
                _norm_content_category
            )
            df["clean_content_category"] = df["clean_content_category"].fillna(
                "unknown"
            )
        else:
            df = df.withColumn(
                "clean_content_category",
                F.when(
                    F.col("content_category").rlike(".*hài kịch.*")
                    | F.col("content_category").rlike(".*hài.*")
                    | F.col("content_category").rlike(".*hước.*"),
                    "hai",
                )
                .when(
                    F.col("content_category").rlike(".*Kinh dị.*")
                    | F.col("content_category").rlike(".*kinh dị.*")
                    | F.col("content_category").rlike(".*hình sự.*"),
                    "hinhsu_kinhdi",
                )
                .when(
                    F.col("content_category").like(" Thiếu nhi")
                    | F.col("content_category").rlike(".*teen.*")
                    | F.col("content_category").rlike(".*Bé.*")
                    | F.col("content_category").rlike(".*thiếu nhi.*")
                    | F.col("content_category").rlike(".*gia đình.*"),
                    "kid_giadinh",
                )
                .when(
                    F.col("content_category").rlike(".*cải lương.*")
                    | F.col("content_category").rlike(".*nhạc.*"),
                    "nhac_tt",
                )
                .when(
                    F.col("content_category").rlike(".*edm.*")
                    | F.col("content_category").rlike(".*hiphop.*")
                    | F.col("content_category").rlike(".*kpop.*"),
                    "nhac_hd",
                )
                .when(
                    F.col("content_category").rlike(".*yoga.*")
                    | F.col("content_category").rlike(".*trang điểm.*")
                    | F.col("content_category").rlike(".*đẹp.*")
                    | F.col("content_category").rlike(".*thẩm mỹ.*")
                    | F.col("content_category").rlike(".*sức khỏe.*")
                    | F.col("content_category").rlike(".*chăm sóc.*"),
                    "suckhoe",
                )
                .when(
                    F.col("content_category").rlike(".*24h.*")
                    | F.col("content_category").rlike(".*thời cuộc.*")
                    | F.col("content_category").rlike(".*thời sự.*")
                    | F.col("content_category").rlike(".*tin.*"),
                    "tintuc",
                )
                .when(
                    F.col("content_category").rlike(".*pool.*")
                    | F.col("content_category").rlike(".*u20.*")
                    | F.col("content_category").rlike(".*olympic.*")
                    | F.col("content_category").rlike(".*Đô vật.*")
                    | F.col("content_category").rlike(".*võ.*")
                    | F.col("content_category").rlike(".*bình luận.*")
                    | F.col("content_category").rlike(".*esport.*")
                    | F.col("content_category").rlike(".*cầu lông.*")
                    | F.col("content_category").rlike(".*f1.*")
                    | F.col("content_category").rlike(".*thể thao.*")
                    | F.col("content_category").rlike(".*tennis.*")
                    | F.col("content_category").rlike(".*bóng.*")
                    | F.col("content_category").rlike(".*quyền anh.*")
                    | F.col("content_category").rlike(".*chung kết.*")
                    | F.col("content_category").rlike(".*vòng Loại.*")
                    | F.col("content_category").rlike(".*cup.*"),
                    "thethao",
                )
                .when(
                    F.col("content_category").rlike(".*tiếng anh.*")
                    | F.col("content_category").rlike(".*chinh phục.*")
                    | F.col("content_category").rlike(".*lớp.*")
                    | F.col("content_category").rlike(".*bài học.*")
                    | F.col("content_category").rlike(".*tư duy.*")
                    | F.col("content_category").rlike(".*toeic.*")
                    | F.col("content_category").rlike(".*ielts.*")
                    | F.col("content_category").rlike(".*tiếng Anh.*")
                    | F.col("content_category").rlike(".*du học.*")
                    | F.col("content_category").rlike(".*ngữ pháp.*"),
                    "giaoduc",
                )
                .when(
                    F.col("content_category").rlike(".*hollywood.*")
                    | F.col("content_category").rlike(".*rạp.*")
                    | F.col("content_category").rlike(".*galaxy.*")
                    | F.col("content_category").rlike(".*hbo.*")
                    | F.col("content_category").rlike(".*galaxy.*"),
                    "traphi",
                )
                .when(
                    F.col("content_category").rlike(".*GameShow.*")
                    | F.col("content_category").rlike(".*chương trình.*")
                    | F.col("content_category").rlike(".*gameShow.*")
                    | F.col("content_category").rlike(".*tài liệu.*"),
                    "gameshow",
                )
                .otherwise("others"),
            )
            df = df.na.fill(
                {
                    "clean_content_category": "unknown",
                }
            )
        return df

    def preprocess_content_parent_type(self, df):
        if self.process_lib in ["pandas"]:
            content_type_df = self.content_type_df.rename(
                columns={"mapping": "content_parent_type"}
            )
            df = df.merge(content_type_df, on="content_type", how="left")
            df.loc[
                (df["content_single"] == 2) & (df["content_parent_type"] == "movie"),
                "content_parent_type",
            ] = "tv_series"
            df["content_parent_type"] = df["content_parent_type"].fillna("unknown")
        else:
            content_type_df = self.content_type_df.withColumnRenamed(
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
        spare_feature_info = conf.DhashSpareFeatureInfo()
        output_feature_names = ["hashed_item_id"]
        hash_dependency_info = {"hashed_item_id": "item_id"}
        df = self.preprocess_hashed_id(
            df,
            output_feature_names,
            hash_dependency_info,
            spare_feature_info,
        )
        output_feature_names = ["hashed_item_id_v2"]
        hash_dependency_info = {"hashed_item_id_v2": "item_id"}
        df = self.preprocess_hashed_id(
            df,
            output_feature_names,
            hash_dependency_info,
            spare_feature_info,
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

    def save_preprocessed_data(self, df):
        save_path = self.save_data_dir / f"{self.save_filename}.parquet"
        save_parquet_data(
            df,
            save_path=save_path,
            partition_cols=["filename_date"],
            process_lib=self.process_lib,
            overwrite=False,
        )

    def run(self):
        save_path = self.save_data_dir / f"{self.save_filename}.parquet"
        logger.info(f"Start preprocess features to {save_path}")
        if self.dates_to_extract is None:
            logger.warning(f"Can not found {save_path}. Loading all data")
        elif len(self.dates_to_extract) == 0:
            logger.info("No new data found. Skip extract features")
            return
        else:
            logger.info(f"Loading raw data from date: {self.dates_to_extract}")
        super().run()


class InteractedFeaturePreprocessing(BaseFeaturePreprocessing):
    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.OBSERVATION_FEATURES,
        data_name_to_get_new_dates=DataName.MOVIE_HISTORY,
    ):
        super().__init__(
            process_lib, raw_data_path, save_filename, data_name_to_get_new_dates
        )
        self.content_type_df = load_parquet_data(
            file_paths=self.raw_data_dir / f"{DataName.CONTENT_TYPE}.parquet",
            process_lib=self.process_lib,
            spark=self.spark,
        )
        self.columns_to_init = [
            "content_id",
            "content_type",
            "username",
            "profile_id",
            "duration",
            "date_time",
            "filename_date",
            "part",
        ]

    def read_processed_data(self):
        movie_df = self._load_raw_data(
            data_name=DataName.MOVIE_HISTORY,
            with_columns=self.columns_to_init,
        )
        vod_df = self._load_raw_data(
            data_name=DataName.VOD_HISTORY,
            with_columns=self.columns_to_init,
        )
        if self.process_lib in ["pandas"]:
            movie_df["is_vod_content"] = False
            vod_df["is_vod_content"] = True
            vod_df.loc[
                vod_df["content_type"] == "21",
                "is_vod_content",
            ] = False
            df = pd.concat([movie_df, vod_df], ignore_index=True).drop_duplicates()
        else:
            movie_df = movie_df.withColumn("is_vod_content", F.lit(False))
            vod_df = vod_df.withColumn("is_vod_content", F.lit(True))
            vod_df = vod_df.withColumn(
                "is_vod_content",
                F.when((vod_df["content_type"] == "21"), False).otherwise(
                    vod_df["is_vod_content"]
                ),
            )
            df = movie_df.union(vod_df)
        return df

    def initialize_dataframe(self, big_df):
        big_df = self.create_user_key(big_df)
        big_df = self.create_item_key(big_df)

        if self.process_lib == "pandas":
            big_df = big_df[
                big_df["content_type"].isin(self.content_type_df["content_type"])
            ]
            big_df = big_df.groupby(
                [
                    "user_id",
                    "item_id",
                    "username",
                    "profile_id",
                    "content_id",
                    "content_type",
                    "filename_date",
                ],
                as_index=False,
            ).agg(
                duration=("duration", "sum"),
                is_vod_content=("is_vod_content", "max"),
            )
            big_df["date_time"] = pd.to_datetime(
                big_df["filename_date"], format=conf.FILENAME_DATE_FORMAT
            )
            # negative sampling
            logger.info("Negative sampling.")
            big_df = self._negative_sample(big_df)
        else:
            big_df = big_df.join(
                self.content_type_df.select("content_type"),
                on="content_type",
                how="inner",
            )
            big_df = big_df.filter(F.col("profile_id") != 0)
            big_df = big_df.groupBy(
                "user_id",
                "item_id",
                "username",
                "profile_id",
                "content_id",
                "content_type",
                "filename_date",
            ).agg(
                F.sum("duration").alias("duration"),
                F.max("is_vod_content").alias("is_vod_content"),
            )
            big_df = big_df.withColumn(
                "date_time",
                F.to_date(F.col("filename_date").cast(StringType()), "yyyyMMdd"),
            )
            # negative sampling
            logger.info("Negative sampling.")
            big_df = self._negative_sample(big_df)
            big_df.persist(storageLevel=StorageLevel.MEMORY_ONLY)
            big_df = big_df.checkpoint()
        logger.info("Negative sampling...done!")
        return big_df

    def _negative_sample(self, big_df):
        negative_sample_ratio = 12
        if self.process_lib == "pandas":
            mean_samples_per_day = (
                big_df.groupby(
                    [
                        "user_id",
                        "username",
                        "profile_id",
                        "is_vod_content",
                        "filename_date",
                    ]
                )["item_id"]
                .count()
                .mean()
            )
            negative_samples_per_day = int(mean_samples_per_day * negative_sample_ratio)
            item_df = big_df[
                [
                    "item_id",
                    "content_id",
                    "content_type",
                    "filename_date",
                    "is_vod_content",
                ]
            ].drop_duplicates()
            user_df = big_df[
                ["user_id", "username", "profile_id", "filename_date"]
            ].drop_duplicates()
            neg_interact_df = self._negative_sample_each_day(
                user_df, item_df, negative_samples_per_day, big_df
            )
            neg_interact_df = neg_interact_df[big_df.columns]
            return pd.concat([big_df, neg_interact_df], ignore_index=True)
        else:
            mean_samples_per_day = (
                big_df.groupby(["user_id", "username", "profile_id", "filename_date"])
                .agg(F.count("item_id").alias("count"))
                .agg(F.mean("count").alias("mean"))
                .select("mean")
                .first()[0]
            )
            negative_samples_per_day = int(mean_samples_per_day * negative_sample_ratio)
            item_df = big_df.select(
                "item_id",
                "content_id",
                "content_type",
                "filename_date",
                "is_vod_content",
            ).dropDuplicates()
            user_df = big_df.select(
                "user_id", "username", "profile_id", "filename_date"
            ).dropDuplicates()
            neg_interact_df = self._negative_sample_each_day(
                user_df, item_df, negative_samples_per_day, big_df
            )
            neg_interact_df = neg_interact_df.select(big_df.columns)
            return big_df.union(neg_interact_df)

    def _negative_sample_each_day(
        self,
        user_df: Union[pd.DataFrame, DataFrame],
        item_df: Union[pd.DataFrame, DataFrame],
        num_negative_samples: int,
        big_df: Union[pd.DataFrame, DataFrame],
    ) -> Union[pd.DataFrame, DataFrame]:
        if self.process_lib == "pandas":
            assert isinstance(user_df, pd.DataFrame)
            assert isinstance(item_df, pd.DataFrame)
            user_df = user_df.set_index("filename_date")
            item_df = item_df.set_index("filename_date")
            filename_dates = user_df.index.unique()
            date_batch_size = 5
            filename_dates_batches = split_batches(filename_dates, date_batch_size)
            neg_interact_dfs = []
            for filename_date in tqdm(filename_dates_batches):
                sub_user_df = user_df.loc[filename_date].reset_index()
                sub_item_df = item_df.loc[filename_date].reset_index()
                neg_interact_df = sub_user_df.merge(
                    sub_item_df, how="inner", on="filename_date"
                )
                neg_interact_df = neg_interact_df.groupby(
                    [
                        "user_id",
                        "username",
                        "profile_id",
                        "filename_date",
                        "is_vod_content",
                    ],
                    as_index=False,
                ).sample(n=num_negative_samples, random_state=42, replace=True)
                neg_interact_df = neg_interact_df.drop_duplicates()
                neg_interact_df["duration"] = 0
                neg_interact_df["date_time"] = pd.to_datetime(
                    neg_interact_df["filename_date"], format=conf.FILENAME_DATE_FORMAT
                )
                neg_interact_dfs.append(neg_interact_df)
            return pd.concat(neg_interact_dfs, ignore_index=True)
        else:
            neg_interact_df = user_df.join(item_df, on="filename_date", how="inner")
            # global sampling to reduce sampling pool
            # before perform stratified sampling
            mean_possible_samples_per_day = (
                big_df.groupby(["filename_date"])
                .count()
                .agg(F.mean("count").alias("mean"))
                .select("mean")
                .first()[0]
            )
            # 1000 times is big enough to maintain result of stratified sampling
            reduced_pool_size = 1000 * num_negative_samples
            sampling_fraction = reduced_pool_size / mean_possible_samples_per_day
            if sampling_fraction < 1:
                neg_interact_df = neg_interact_df.sample(
                    fraction=sampling_fraction, seed=40
                )
            # draw (num_negative_samples) sample
            # for each user-date pair from smaller pool
            neg_interact_df = neg_interact_df.withColumn(
                "random_group", F.floor(F.rand(seed=42) * num_negative_samples)
            ).withColumn("random_selection", F.rand(seed=41))
            # todo: replace F.max_by("content", "random_selection")
            # by F.any_value("content") when pyspark 3.5.0 is available
            neg_interact_df = (
                neg_interact_df.groupby(
                    [
                        "user_id",
                        "username",
                        "profile_id",
                        "filename_date",
                        "is_vod_content",
                        "random_group",
                    ]
                ).agg(F.max_by("item_id", "random_selection").alias("item_id"))
            ).drop("random_group", "random_selection")
            neg_interact_df = neg_interact_df.withColumn(
                "content_type", F.split(F.col("item_id"), "#", 2)[0]
            ).withColumn("content_id", F.split(F.col("item_id"), "#", 2)[1])
            neg_interact_df = neg_interact_df.withColumn(
                "date_time",
                F.to_date(F.col("filename_date").cast("string"), "yyyyMMdd"),
            ).withColumn("duration", F.lit(0))
            return neg_interact_df

    def preprocess_feature(self, df):
        if self.process_lib in ["pandas"]:
            df["is_interacted"] = 2
            df.loc[
                (df["duration"] < conf.VOD_DIRTY_CLICK_DURATION_THRESHOLD)
                & df["is_vod_content"],
                "is_interacted",
            ] = 0
            df.loc[
                (df["duration"] < conf.MOVIE_DIRTY_CLICK_DURATION_THRESHOLD)
                & ~df["is_vod_content"],
                "is_interacted",
            ] = 0
            df.loc[df["duration"] == 0, "is_interacted"] = 1
        else:
            df = df.withColumn("is_interacted", F.lit(2))
            df = df.withColumn(
                "is_interacted",
                F.when(
                    (
                        (F.col("duration") < conf.VOD_DIRTY_CLICK_DURATION_THRESHOLD)
                        & F.col("is_vod_content")
                    )
                    | (
                        (F.col("duration") < conf.MOVIE_DIRTY_CLICK_DURATION_THRESHOLD)
                        & ~F.col("is_vod_content")
                    ),
                    F.lit(0),
                ).otherwise(F.col("is_interacted")),
            )
            df = df.withColumn(
                "is_interacted",
                F.when(F.col("duration") == 0, F.lit(1)).otherwise(
                    F.col("is_interacted")
                ),
            )
        return df

    def save_preprocessed_data(self, df):
        save_path = self.save_data_dir / f"{self.save_filename}.parquet"
        save_parquet_data(
            df,
            save_path=save_path,
            partition_cols=["filename_date"],
            process_lib=self.process_lib,
            overwrite=False,
        )

    def run(self):
        save_path = self.save_data_dir / f"{self.save_filename}.parquet"
        logger.info(f"Start preprocess features to {save_path}")
        if self.dates_to_extract is None:
            logger.warning(f"Can not found {save_path}. Loading all data")
        elif len(self.dates_to_extract) == 0:
            logger.info("No new data found. Skip extract features")
            return
        else:
            logger.info(f"Loading raw data from date: {self.dates_to_extract}")
        super().run()


class BaseOnlineFeaturePreprocessing(BaseFeaturePreprocessing):
    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename="",
        data_name_to_get_new_dates=DataName.MOVIE_HISTORY,
    ):
        super().__init__(
            process_lib, raw_data_path, save_filename, data_name_to_get_new_dates
        )
        self.columns_to_init = [
            "content_id",
            "content_type",
            "username",
            "profile_id",
            "duration",
            "date_time",
            "filename_date",
            "part",
        ]

    def read_processed_data(self):
        movie_df = self._load_raw_data(
            data_name=DataName.MOVIE_HISTORY,
            with_columns=self.columns_to_init,
        )
        vod_df = self._load_raw_data(
            data_name=DataName.VOD_HISTORY,
            with_columns=self.columns_to_init,
        )
        if self.process_lib == "pandas":
            df = pd.concat([movie_df, vod_df])
        else:
            df = movie_df.union(vod_df)
        return df

    def initialize_dataframe(self, df):
        df = self.create_user_key(df)
        df = self.create_item_key(df)
        return df

    def save_preprocessed_data(self, df):
        save_path = self.save_data_dir / f"{self.save_filename}.parquet"
        save_parquet_data(
            df,
            save_path=save_path,
            partition_cols=["filename_date"],
            process_lib=self.process_lib,
            overwrite=False,
        )

    def run(self):
        save_path = self.save_data_dir / f"{self.save_filename}.parquet"
        logger.info(f"Start preprocess features to {save_path}")
        if self.dates_to_extract is None:
            logger.warning(f"Can not found {save_path}. Loading all data")
        elif len(self.dates_to_extract) == 0:
            logger.info("No new data found. Skip extract features")
            return
        else:
            logger.info(f"Loading raw data from date: {self.dates_to_extract}")
        super().run()


class OnlineItemFeaturePreprocessing(BaseOnlineFeaturePreprocessing):
    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.ONLINE_ITEM_FEATURES,
        data_name_to_get_new_dates=DataName.MOVIE_HISTORY,
    ):
        super().__init__(
            process_lib, raw_data_path, save_filename, data_name_to_get_new_dates
        )

    def preprocess_feature(self, big_df):
        if self.process_lib in ["pandas"]:
            big_df = big_df[big_df["content_type"].astype(str) != "31"]
            big_df = big_df[
                ["profile_id", "content_id", "content_type", "filename_date"]
            ].drop_duplicates()
            big_df = big_df.reset_index(drop=True)
            big_df["item_id"] = big_df["content_type"] + "#" + big_df["content_id"]

            for i, p_date in enumerate(self.dates_to_extract):
                begin_date = get_date_before(p_date, num_days_before=30)
                df_small = big_df[
                    (big_df.filename_date <= p_date)
                    & (big_df.filename_date > begin_date)
                ]
                if df_small["filename_date"].drop_duplicates().count() < 15:
                    df_small = df_small["item_id"].drop_duplicates().to_frame()
                    df_small["count"] = 0
                    df_small["popularity_item_group"] = "others"
                else:
                    df_small = df_small.groupby("item_id").size().to_frame("count")
                    df_small = df_small.sort_values(by=["count"], ascending=False)
                    df_small = df_small.reset_index().reset_index()
                    df_small["popularity_item_group"] = ">2000"
                    df_small.loc[
                        df_small.index < 2000, "popularity_item_group"
                    ] = "1001-2000"
                    df_small.loc[
                        df_small.index < 1000, "popularity_item_group"
                    ] = "301-1000"
                    df_small.loc[
                        df_small.index < 300, "popularity_item_group"
                    ] = "101-300"
                    df_small.loc[df_small.index < 100, "popularity_item_group"] = "100"
                    df_small.drop(columns=["index"], inplace=True)

                df_small["filename_date"] = p_date
                df_small["date_time"] = pd.to_datetime(
                    df_small["filename_date"], format=conf.FILENAME_DATE_FORMAT
                )
                if i == 0:
                    popular_item_group = df_small
                else:
                    popular_item_group = pd.concat([popular_item_group, df_small])
            popular_item_group = popular_item_group.reset_index(drop=True)
        else:
            big_df = big_df.filter(F.col("content_type").cast(StringType()) != "31")
            big_df = big_df.select(
                "profile_id", "item_id", "filename_date"
            ).dropDuplicates()
            popular_item_group = self.spark.createDataFrame([], StructType([]))

            for i, p_date in enumerate(self.dates_to_extract):
                begin_date = get_date_before(p_date, num_days_before=30)
                df_small = big_df.filter(
                    (big_df.filename_date <= p_date)
                    & (big_df.filename_date > begin_date)
                )
                if df_small.select("filename_date").distinct().count() < 15:
                    df_small = df_small.select("item_id").distinct()
                    df_small = df_small.withColumn("count", F.lit(0))
                    df_small = df_small.withColumn(
                        "popularity_item_group", F.lit("others")
                    )
                else:
                    df_small = df_small.groupBy("item_id").count()
                    df_small = df_small.withColumn(
                        "row",
                        F.row_number().over(
                            Window.partitionBy(F.lit("1")).orderBy(
                                F.col("count").desc(),
                            )
                        ),
                    )
                    df_small = df_small.withColumn(
                        "popularity_item_group",
                        F.when(F.col("row") <= 100, F.lit("100"))
                        .when(F.col("row") <= 300, F.lit("101-300"))
                        .when(F.col("row") <= 1000, F.lit("301-1000"))
                        .when(F.col("row") <= 2000, F.lit("1001-2000"))
                        .otherwise(F.lit(">2000")),
                    )
                    df_small = df_small.drop("row")
                df_small = df_small.withColumn("filename_date", F.lit(p_date))
                df_small = df_small.withColumn(
                    "date_time",
                    F.to_date(F.col("filename_date").cast(StringType()), "yyyyMMdd"),
                )
                if i == 0:
                    popular_item_group = df_small
                else:
                    popular_item_group = popular_item_group.union(df_small)

        return popular_item_group


class OnlineUserFeaturePreprocessing(BaseOnlineFeaturePreprocessing):
    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.ONLINE_USER_FEATURES,
        data_name_to_get_new_dates=DataName.MOVIE_HISTORY,
    ):
        super().__init__(
            process_lib, raw_data_path, save_filename, data_name_to_get_new_dates
        )

    def preprocess_feature(self, big_df):
        if self.process_lib in ["pandas"]:
            big_df = big_df[big_df["content_type"].astype(str) != "31"]
            big_df = big_df[
                ["user_id", "content_type", "filename_date"]
            ].drop_duplicates()
            big_df = big_df.reset_index(drop=True)
            big_df["movie_or_vod"] = "vod"
            big_df.loc[
                big_df["content_type"].isin(conf.MOVIE_TYPE_GROUP), "movie_or_vod"
            ] = "movie"

            for i, p_date in enumerate(self.dates_to_extract):
                begin_date = get_date_before(
                    p_date, num_days_before=conf.ROLLING_PERIOD_FOR_PREFER_TYPE
                )
                df_small = big_df[
                    (big_df.filename_date <= p_date)
                    & (big_df.filename_date > begin_date)
                ]
                if df_small["filename_date"].drop_duplicates().count() < int(
                    conf.ROLLING_PERIOD_FOR_PREFER_TYPE / 2
                ):
                    df_small = df_small["user_id"].drop_duplicates().to_frame()
                    df_small["prefer_movie_type"] = "0"
                    df_small["prefer_vod_type"] = "0"
                else:
                    df_small = (
                        df_small.groupby(["user_id", "movie_or_vod"])
                        .agg({"content_type": "count"})
                        .reset_index()
                    )
                    df_small = df_small.rename(columns={"content_type": "prefer_type"})
                    df_small = df_small.pivot(
                        index="user_id", columns="movie_or_vod", values="prefer_type"
                    ).reset_index()
                    df_small["movie"] = df_small["movie"].fillna("0")
                    df_small["vod"] = df_small["vod"].fillna("0")
                    df_small["movie"] = df_small["movie"].astype(int).astype(str)
                    df_small["vod"] = df_small["vod"].astype(int).astype(str)
                    df_small = df_small.rename(
                        columns={"movie": "prefer_movie_type", "vod": "prefer_vod_type"}
                    )
                    df_small["prefer_movie_type"] = df_small[
                        "prefer_movie_type"
                    ].fillna("0")
                    df_small["prefer_vod_type"] = df_small["prefer_vod_type"].fillna(
                        "0"
                    )

                df_small["filename_date"] = p_date
                df_small["date_time"] = pd.to_datetime(
                    df_small["filename_date"], format=conf.FILENAME_DATE_FORMAT
                )
                if i == 0:
                    user_prefer_type = df_small
                else:
                    user_prefer_type = pd.concat([user_prefer_type, df_small])
            user_prefer_type = user_prefer_type.reset_index(drop=True)
        else:
            big_df = big_df.filter(F.col("content_type").cast(StringType()) != "31")
            big_df = big_df.select(
                "user_id", "content_type", "filename_date"
            ).dropDuplicates()
            big_df = big_df.withColumn(
                "movie_or_vod",
                F.when(
                    F.col("content_type").isin(conf.MOVIE_TYPE_GROUP), F.lit("movie")
                ).otherwise(F.lit("vod")),
            )
            user_prefer_type = self.spark.createDataFrame([], StructType([]))

            for i, p_date in enumerate(self.dates_to_extract):
                begin_date = get_date_before(
                    p_date, num_days_before=conf.ROLLING_PERIOD_FOR_PREFER_TYPE
                )
                df_small = big_df.filter(
                    (big_df.filename_date <= p_date)
                    & (big_df.filename_date > begin_date)
                )
                if df_small.select("filename_date").distinct().count() < int(
                    conf.ROLLING_PERIOD_FOR_PREFER_TYPE / 2
                ):
                    df_small = df_small.select("user_id").distinct()
                    df_small = df_small.withColumn("prefer_movie_type", F.lit("0"))
                    df_small = df_small.withColumn("prefer_vod_type", F.lit("0"))
                else:
                    df_small = df_small.groupBy("user_id", "movie_or_vod").agg(
                        F.count("content_type").alias("prefer_type")
                    )
                    df_small = (
                        df_small.groupBy("user_id")
                        .pivot("movie_or_vod")
                        .agg(F.first("prefer_type"))
                    )
                    df_small = df_small.withColumnRenamed(
                        "movie", "prefer_movie_type"
                    ).withColumnRenamed("vod", "prefer_vod_type")
                    df_small = df_small.na.fill(
                        {"prefer_movie_type": "0", "prefer_vod_type": "0"}
                    )
                df_small = df_small.withColumn("filename_date", F.lit(p_date))
                df_small = df_small.withColumn(
                    "date_time",
                    F.to_date(F.col("filename_date").cast(StringType()), "yyyyMMdd"),
                )
                if i == 0:
                    user_prefer_type = df_small
                else:
                    user_prefer_type = user_prefer_type.union(df_small)

        return user_prefer_type


def _norm_content_category(item):
    item = str(item)
    if (
        ("hài kịch" in item.lower())
        | ("hài" in item.lower())
        | ("hước" in item.lower())
    ):
        return "hai"
    elif (
        ("Kinh dị" in item) | ("kinh dị" in item.lower()) | ("hình sự" in item.lower())
    ):
        return "hinhsu_kinhdi"
    elif (
        (" Thiếu nhi" == item)
        | ("teen" in item.lower())
        | ("Bé" in item)
        | ("thiếu nhi" in item.lower())
        | ("gia đình" in item.lower())
    ):
        return "kid_giadinh"
    elif ("cải lương" in item.lower()) | ("nhạc" in item.lower()):
        return "nhac_tt"
    elif (
        ("edm" in item.lower()) | ("hiphop" in item.lower()) | ("kpop" in item.lower())
    ):
        return "nhac_hd"
    elif (
        ("yoga" in item.lower())
        | ("trang điểm" in item.lower())
        | ("đẹp" in item.lower())
        | ("thẩm mỹ" in item.lower())
        | ("sức khỏe" in item.lower())
        | ("chăm sóc" in item.lower())
    ):
        return "suckhoe"
    elif (
        ("24h" in item.lower())
        | ("thời cuộc" in item.lower())
        | ("thời sự" in item.lower())
        | ("tin" in item.lower())
    ):
        return "tintuc"
    elif (
        ("pool".lower() in item.lower())
        | ("u20".lower() in item.lower())
        | ("olympic".lower() in item.lower())
        | ("Đô vật".lower() in item.lower())
        | ("võ" in item.lower())
        | ("bình luận" in item.lower())
        | ("esport" in item.lower())
        | ("cầu lông" in item.lower())
        | ("f1" in item.lower())
        | ("thể thao" in item.lower())
        | ("tennis" in item.lower())
        | ("bóng" in item.lower())
        | ("quyền anh" in item.lower())
        | ("chung kết" in item.lower())
        | ("vòng Loại" in item.lower())
        | ("cup" in item.lower())
    ):
        return "thethao"
    elif (
        ("tiếng anh" in item.lower())
        | ("chinh phục" in item.lower())
        | ("lớp" in item.lower())
        | ("bài học" in item.lower())
        | ("tư duy" in item.lower())
        | ("toeic" in item.lower())
        | ("ielts" in item.lower())
        | ("tiếng Anh" in item.lower())
        | ("du học" in item.lower())
        | ("ngữ pháp" in item.lower())
    ):
        return "giaoduc"
    elif (
        ("hollywood" in item.lower())
        | ("rạp" in item.lower())
        | ("galaxy" in item.lower())
        | ("hbo" in item.lower())
        | ("độc quyền" in item.lower())
    ):
        return "traphi"
    elif (
        ("GameShow" in item)
        | ("chương trình" in item.lower())
        | ("gameShow" in item.lower())
        | ("tài liệu" in item.lower())
    ):
        return "gameshow"
    else:
        return "others"


if __name__ == "__main__":
    UserFeaturePreprocessing().run()
    ABUserFeaturePreprocessing().run()
    ContentFeaturePreprocessing().run()
    InteractedFeaturePreprocessing().run()
    OnlineItemFeaturePreprocessing().run()
    OnlineUserFeaturePreprocessing().run()
