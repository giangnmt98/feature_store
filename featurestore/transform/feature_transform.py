from feathr import (
    BOOLEAN,
    FLOAT,
    INT32,
    INT64,
    STRING,
    DerivedFeature,
    Feature,
    WindowAggTransformation,
)

from configs.conf import (
    DATALOADER_BATCH_NUMBER,
    DURATION_THRESHOLD_FOR_WEIGHTED_LR,
    HASHED_CONTENT_CATEGORY_BS,
    HASHED_PUBLISH_YEAR_BIWEEKLY_BS,
    INFERRING_USER_WEIGHT,
    NUMBER_OF_RANDOM_USER_GROUP,
    POSITIVE_WEIGH_FOR_POPULARITY_GROUP,
    SpareFeatureInfo,
)
from featurestore.transform.key_definition import KeyDefinition


class FeatureTransform:
    def __init__(self):
        self.key_collection = KeyDefinition().key_collection
        self.user_id = self.key_collection["user_id"]
        self.item_id = self.key_collection["item_id"]

        # ANCHORED FEATURE
        # user features
        self.f_age_group = Feature(
            name="user_age_group",
            key=self.user_id,
            feature_type=STRING,
            transform="if(isnull(age_group), 'married', age_group)",
        )
        self.f_province = Feature(
            name="user_province",
            key=self.user_id,
            feature_type=STRING,
            transform="if(isnull(province), 'HCM', province)",
        )
        self.f_package_code = Feature(
            name="user_package_code",
            key=self.user_id,
            feature_type=STRING,
            transform="if(isnull(package_code), 'MYTV021', package_code)",
        )
        self.f_sex = Feature(
            name="user_sex",
            key=self.user_id,
            feature_type=STRING,
            transform="if(isnull(sex), '1', sex)",
        )
        self.f_platform = Feature(
            name="user_platform",
            key=self.user_id,
            feature_type=STRING,
            transform="if(isnull(platform), 'b2c-android', platform)",
        )
        self.f_hashed_user_id = Feature(
            name="hashed_user_id",
            key=self.user_id,
            feature_type=INT64,
            transform="hashed_user_id",
        )
        self.f_hashed_user_id_v2 = Feature(
            name="hashed_user_id_v2",
            key=self.user_id,
            feature_type=INT64,
            transform="hashed_user_id_v2",
        )

        # content features
        self.f_clean_content_country = Feature(
            name="clean_content_country",
            key=self.item_id,
            feature_type=STRING,
            transform="clean_content_country",
        )
        self.f_clean_content_category = Feature(
            name="clean_content_category",
            key=self.item_id,
            feature_type=STRING,
            transform="clean_content_category",
        )
        self.f_content_category = Feature(
            name="content_category",
            key=self.item_id,
            feature_type=STRING,
            transform="content_category",
        )
        self.f_publish_month = Feature(
            name="publish_month",
            key=self.item_id,
            feature_type=STRING,
            transform="date_format(date_trunc('DD', publish_date), 'yyyy-MM')",
        )
        self.f_publish_week = Feature(
            name="publish_week",
            key=self.item_id,
            feature_type=STRING,
            transform="concat("
            "cast(year(date_trunc('DD', publish_date)) as string), "
            "'_', "
            "(cast(weekofyear(date_trunc('DD', publish_date)) as string)))",
        )
        self.f_publish_year_biweekly = Feature(
            name="publish_year_biweekly",
            key=self.item_id,
            feature_type=STRING,
            transform="format_string('%d_%02d', "
            "year(date_trunc('DD', publish_date)), "
            "cast((weekofyear(date_trunc('DD', publish_date))/2) as int))",
        )
        self.f_hashed_item_id = Feature(
            name="hashed_item_id",
            key=self.item_id,
            feature_type=INT64,
            transform="hashed_item_id",
        )
        self.f_hashed_item_id_v2 = Feature(
            name="hashed_item_id_v2",
            key=self.item_id,
            feature_type=INT64,
            transform="hashed_item_id_v2",
        )
        self.f_content_parent_type = Feature(
            name="content_parent_type",
            key=self.item_id,
            feature_type=STRING,
            transform="content_parent_type",
        )
        self.f_is_vod_content = Feature(
            name="item_is_vod_content",
            key=self.item_id,
            feature_type=BOOLEAN,
            transform="is_vod_content",
        )
        self.f_is_movie_content = Feature(
            name="item_is_movie_content",
            key=self.item_id,
            feature_type=BOOLEAN,
            transform="is_movie_content",
        )
        self.f_is_channel_content = Feature(
            name="item_is_channel_content",
            key=self.item_id,
            feature_type=BOOLEAN,
            transform="is_channel_content",
        )

        # online item features
        self.f_popularity_item_group = Feature(
            name="popularity_item_group",
            key=self.item_id,
            feature_type=STRING,
            transform=WindowAggTransformation(
                agg_expr="popularity_item_group", agg_func="LATEST", window="7d"
            ),
        )

        # online user features
        self.f_prefer_movie_type = Feature(
            name="prefer_movie_type",
            key=self.user_id,
            feature_type=STRING,
            transform=WindowAggTransformation(
                agg_expr="prefer_movie_type", agg_func="LATEST", window="7d"
            ),
        )
        self.f_prefer_vod_type = Feature(
            name="prefer_vod_type",
            key=self.user_id,
            feature_type=STRING,
            transform=WindowAggTransformation(
                agg_expr="prefer_vod_type", agg_func="LATEST", window="7d"
            ),
        )

        self.f_is_weekend = Feature(
            name="is_weekend",
            feature_type=INT32,
            transform="if("
            "(dayofweek(to_date(date_time,'yyyy-MM-dd'))-1) in(0,6), "
            "1, 0)",
        )
        self.f_ab_group_id = Feature(
            name="ab_group_id",
            key=self.user_id,
            feature_type=INT32,
            transform="group_id",
        )

        cond_map_content_type = encoded_func_string(
            SpareFeatureInfo.encoded_features["encoded_content_type"], "content_type", 0
        )
        self.f_encode_content_type = Feature(
            name="encoded_content_type",
            key=self.item_id,
            feature_type=INT32,
            transform=cond_map_content_type,
        )

        hash_content_category_str = hashed_func_string(
            "content_category", HASHED_CONTENT_CATEGORY_BS
        )
        self.f_hashed_content_category = Feature(
            name="hashed_content_category",
            key=self.item_id,
            feature_type=INT64,
            transform=hash_content_category_str,
        )
        hash_batch_idx_str = hashed_func_string("profile_id", DATALOADER_BATCH_NUMBER)
        self.f_batch_idx = Feature(
            name="batch_idx",
            feature_type=INT64,
            transform=hash_batch_idx_str,
        )

        # DERIVED FEATURE
        self.f_movie_publish_month = DerivedFeature(
            name="movie_publish_month",
            key=self.item_id,
            feature_type=STRING,
            input_features=[self.f_is_movie_content, self.f_publish_month],
            transform="if(boolean(item_is_movie_content), publish_month, 'not_movie')",
        )
        self.f_movie_publish_week = DerivedFeature(
            name="movie_publish_week",
            key=self.item_id,
            feature_type=STRING,
            input_features=[self.f_is_movie_content, self.f_publish_week],
            transform="if(boolean(item_is_movie_content), publish_week, 'not_movie')",
        )
        self.f_is_infering_user = DerivedFeature(
            name="is_infering_user",
            key=self.user_id,
            feature_type=BOOLEAN,
            input_features=[self.f_ab_group_id],
            transform="if(boolean(ab_group_id=1), True, False)",
        )
        self.f_user_weight = DerivedFeature(
            name="user_weight",
            key=self.user_id,
            feature_type=FLOAT,
            input_features=[self.f_is_infering_user],
            transform=f"if(boolean(is_infering_user), {INFERRING_USER_WEIGHT}, 1)",
        )
        self.f_item_weight = DerivedFeature(
            name="item_weight",
            key=self.item_id,
            feature_type=FLOAT,
            input_features=[self.f_popularity_item_group],
            transform=f"if(boolean(popularity_item_group='>2000'), "
            f"{POSITIVE_WEIGH_FOR_POPULARITY_GROUP[4]}, "
            f"if(boolean(popularity_item_group='1001-2000'), "
            f"{POSITIVE_WEIGH_FOR_POPULARITY_GROUP[3]}, "
            f"if(boolean(popularity_item_group='301-1000'), "
            f"{POSITIVE_WEIGH_FOR_POPULARITY_GROUP[2]}, "
            f"if(boolean(popularity_item_group='101-300'), "
            f"{POSITIVE_WEIGH_FOR_POPULARITY_GROUP[1]}, "
            f"if(boolean(popularity_item_group='100'), "
            f"{POSITIVE_WEIGH_FOR_POPULARITY_GROUP[0]}, "
            f"1)))))",
        )
        self.f_weighted_lr = DerivedFeature(
            name="weighted_lr",
            key=[self.user_id, self.item_id],
            feature_type=FLOAT,
            input_features=[self.f_user_weight, self.f_item_weight],
            transform=f"if(boolean(is_interacted=0), 1, "
            f"if(boolean(is_interacted=1), 1, ("
            f"least(cast(duration as float), {DURATION_THRESHOLD_FOR_WEIGHTED_LR}) "
            f"/ {DURATION_THRESHOLD_FOR_WEIGHTED_LR}"
            f"))) * item_weight * user_weight",
        )

        cond_map_country = encoded_func_string(
            SpareFeatureInfo.encoded_features["encoded_content_country"],
            "clean_content_country",
            0,
        )
        cond_map_content_parent_type = encoded_func_string(
            SpareFeatureInfo.encoded_features["encoded_content_parent_type"],
            "content_parent_type",
            0,
        )
        cond_map_province = encoded_func_string(
            SpareFeatureInfo.encoded_features["encoded_user_province"],
            "user_province",
            0,
        )
        cond_map_package_code = encoded_func_string(
            SpareFeatureInfo.encoded_features["encoded_user_package_code"],
            "user_package_code",
            0,
        )
        cond_map_age_group = encoded_func_string(
            SpareFeatureInfo.encoded_features["encoded_age_group"], "user_age_group", 0
        )
        cond_map_prefer_movie_type = encoded_func_string(
            SpareFeatureInfo.encoded_features["encoded_content_type"],
            "prefer_movie_type",
            0,
        )
        cond_map_prefer_vod_type = encoded_func_string(
            SpareFeatureInfo.encoded_features["encoded_content_type"],
            "prefer_vod_type",
            0,
        )
        self.f_encode_content_country = DerivedFeature(
            name="encoded_content_country",
            key=self.item_id,
            feature_type=INT32,
            input_features=[self.f_clean_content_country],
            transform=cond_map_country,
        )
        self.f_encode_content_parent_type = DerivedFeature(
            name="encoded_content_parent_type",
            key=self.item_id,
            feature_type=INT32,
            input_features=[self.f_content_parent_type],
            transform=cond_map_content_parent_type,
        )
        self.f_encode_province = DerivedFeature(
            name="encoded_user_province",
            key=self.user_id,
            feature_type=INT32,
            input_features=[self.f_province],
            transform=cond_map_province,
        )
        self.f_encode_package_code = DerivedFeature(
            name="encoded_user_package_code",
            key=self.user_id,
            feature_type=INT32,
            input_features=[self.f_package_code],
            transform=cond_map_package_code,
        )
        self.f_encode_age_group = DerivedFeature(
            name="encoded_age_group",
            key=self.user_id,
            feature_type=INT32,
            input_features=[self.f_age_group],
            transform=cond_map_age_group,
        )
        self.f_encode_prefer_movie_type = DerivedFeature(
            name="encoded_prefer_movie_type",
            key=self.user_id,
            feature_type=INT32,
            input_features=[self.f_prefer_movie_type],
            transform=cond_map_prefer_movie_type,
        )
        self.f_encode_prefer_vod_type = DerivedFeature(
            name="encoded_prefer_vod_type",
            key=self.user_id,
            feature_type=INT32,
            input_features=[self.f_prefer_vod_type],
            transform=cond_map_prefer_vod_type,
        )

        hash_publish_year_biweekly_str = hashed_func_string(
            "publish_year_biweekly", HASHED_PUBLISH_YEAR_BIWEEKLY_BS
        )
        hash_random_user_group_str = hashed_func_string(
            "profile_id", NUMBER_OF_RANDOM_USER_GROUP
        )
        self.f_hashed_publish_year_biweekly = DerivedFeature(
            name="hashed_publish_year_biweekly",
            key=self.item_id,
            feature_type=INT64,
            input_features=[self.f_publish_year_biweekly],
            transform=hash_publish_year_biweekly_str,
        )
        self.f_random_user_group = DerivedFeature(
            name="random_user_group",
            key=self.user_id,
            feature_type=INT64,
            input_features=[self.f_is_infering_user],
            transform=f"if(is_infering_user, 0, ({hash_random_user_group_str}+1))",
        )


def encoded_func_string(dictionary, col, default_value):
    cond = ""
    for key, value in dictionary.items():
        cond += f"if(boolean({col}='{key}'), {value}, "
    cond += f"{default_value}" + ")" * len(dictionary)
    return cond


def hashed_func_string(col, bucket_size):
    func_string = (
        f"cast("
        f"conv("
        f"substring("
        f"md5("
        f"cast({col} as string)"
        f"), 1, 15"
        f"), 16, 10"
        f") as long) % {bucket_size}"
    )
    return func_string
