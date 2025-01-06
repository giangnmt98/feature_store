from feathr import INPUT_CONTEXT, FeatureAnchor

from featurestore.base.schemas.registry_config import FeatureRegistryConfig
from featurestore.base.utils.config import parse_registry_config
from featurestore.base.utils.logger import logger
from featurestore.base.utils.utils import return_or_load
from featurestore.transform.feature_transform import FeatureTransform
from featurestore.transform.source_definition import SourceDefinition


class FeatureRegistry:
    def __init__(
        self,
        config_path: str,
        feathr_client,
        raw_data_path: str,
    ):
        self.registry_config = return_or_load(
            config_path, FeatureRegistryConfig, parse_registry_config
        )
        self.client = feathr_client
        self.raw_data_path = raw_data_path + "/" + self.registry_config.raw_data_path
        self.feature_transform = FeatureTransform()

        self.user_batch_source = self._get_user_batch_source()
        self.content_info_batch_source = self._get_content_batch_source()
        self.online_user_batch_source = self._get_online_user_batch_source()
        self.online_item_batch_source = self._get_online_item_batch_source()
        self.ab_user_batch_source = self._get_ab_user_batch_source()

        self.anchored_feature_list = self._create_anchored_feature_list()
        self.derived_feature_list = self._create_derived_feature_list()

    def _get_user_batch_source(self):
        return SourceDefinition(self.raw_data_path).user_batch_source

    def _get_content_batch_source(self):
        return SourceDefinition(self.raw_data_path).content_info_batch_source

    def _get_ab_user_batch_source(self):
        return SourceDefinition(self.raw_data_path).ab_user_batch_source

    def _get_online_item_batch_source(self):
        return SourceDefinition(self.raw_data_path).online_item_batch_source

    def _get_online_user_batch_source(self):
        return SourceDefinition(self.raw_data_path).online_user_batch_source

    def _create_anchored_feature_list(self):
        user_features = [
            self.feature_transform.f_age_group,
            self.feature_transform.f_province,
            self.feature_transform.f_package_code,
            self.feature_transform.f_sex,
            self.feature_transform.f_platform,
        ]
        user_feature_anchor = FeatureAnchor(
            name="userFeatures", source=self.user_batch_source, features=user_features
        )

        profile_features = [
            self.feature_transform.f_hashed_user_id,
            self.feature_transform.f_hashed_user_id_v2,
        ]
        profile_feature_anchor = FeatureAnchor(
            name="profileFeatures",
            source=self.user_batch_source,
            features=profile_features,
        )

        content_features = [
            self.feature_transform.f_clean_content_country,
            self.feature_transform.f_clean_content_category,
            self.feature_transform.f_content_category,
            self.feature_transform.f_publish_month,
            self.feature_transform.f_publish_week,
            self.feature_transform.f_publish_year_biweekly,
            self.feature_transform.f_hashed_content_category,
            self.feature_transform.f_hashed_item_id,
            self.feature_transform.f_hashed_item_id_v2,
            self.feature_transform.f_content_parent_type,
            self.feature_transform.f_is_vod_content,
            self.feature_transform.f_is_movie_content,
            self.feature_transform.f_is_channel_content,
            self.feature_transform.f_encode_content_type,
        ]
        content_feature_anchor = FeatureAnchor(
            name="contentFeatures",
            source=self.content_info_batch_source,
            features=content_features,
        )

        online_item_features = [
            self.feature_transform.f_popularity_item_group,
        ]
        online_item_feature_anchor = FeatureAnchor(
            name="onlineItemFeatures",
            source=self.online_item_batch_source,
            features=online_item_features,
        )

        online_user_features = [
            self.feature_transform.f_prefer_movie_type,
            self.feature_transform.f_prefer_vod_type,
        ]
        online_user_feature_anchor = FeatureAnchor(
            name="onlineUserFeatures",
            source=self.online_user_batch_source,
            features=online_user_features,
        )

        ab_user_features = [
            self.feature_transform.f_ab_group_id,
        ]
        ab_user_feature_anchor = FeatureAnchor(
            name="abUserFeatures",
            source=self.ab_user_batch_source,
            features=ab_user_features,
        )

        context_features = [
            self.feature_transform.f_batch_idx,
            self.feature_transform.f_is_weekend,
        ]
        context_feature_anchor = FeatureAnchor(
            name="contextFeatures",
            source=INPUT_CONTEXT,
            features=context_features,
        )
        return [
            user_feature_anchor,
            profile_feature_anchor,
            content_feature_anchor,
            online_item_feature_anchor,
            online_user_feature_anchor,
            ab_user_feature_anchor,
            context_feature_anchor,
        ]

    def _create_derived_feature_list(self):
        return [
            self.feature_transform.f_movie_publish_month,
            self.feature_transform.f_movie_publish_week,
            self.feature_transform.f_is_infering_user,
            self.feature_transform.f_user_weight,
            self.feature_transform.f_item_weight,
            self.feature_transform.f_weighted_lr,
            self.feature_transform.f_encode_content_country,
            self.feature_transform.f_encode_content_parent_type,
            self.feature_transform.f_encode_province,
            self.feature_transform.f_encode_package_code,
            self.feature_transform.f_encode_age_group,
            self.feature_transform.f_hashed_publish_year_biweekly,
            self.feature_transform.f_random_user_group,
            self.feature_transform.f_encode_prefer_movie_type,
            self.feature_transform.f_encode_prefer_vod_type,
        ]

    def _build_features(self):
        self.client.build_features(
            anchor_list=self.anchored_feature_list,
            derived_feature_list=self.derived_feature_list,
        )

    def _register_features(self):
        try:
            self.client.register_features()
        except Exception as e:
            logger.warning(e)

    def run(self):
        self._build_features()
        # self._register_features()
