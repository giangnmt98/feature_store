"""
Module: feature_registry

This module defines the `FeatureRegistry` class, which is responsible for managing and
registering features into a feature store. It processes feature definitions, groups them
into anchors, creates derived features, and facilitates building or registering these
features into the store.
"""

from feathr import INPUT_CONTEXT, FeatureAnchor

from featurestore.base.schemas.registry_config import FeatureRegistryConfig
from featurestore.base.utils.config import parse_registry_config
from featurestore.base.utils.logger import logger
from featurestore.base.utils.utils import return_or_load
from featurestore.transform.feature_transform import FeatureTransform
from featurestore.transform.source_definition import SourceDefinition


class FeatureRegistry:
    """
    FeatureRegistry is responsible for managing and registering features in the
    feature store.

    This registry processes feature definitions, groups them into anchors,
    creates derived features, and builds or registers them into the feature store.

    Attributes:
        registry_config (FeatureRegistryConfig): Parsed configuration object containing
        details about the registry.
        client: Feathr client for interacting with the feature store.
        raw_data_path (str): Path to the raw data directory for defining batch sources.
        feature_transform (FeatureTransform): Contains the transformation logic for
        feature creation.
        user_batch_source, content_info_batch_source, ab_user_batch_source,
        online_user_batch_source, online_item_batch_source: Batch sources for different
        categories of features.
        anchored_feature_list (list): List of feature anchors.
        derived_feature_list (list): List of derived features.
    """

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
        """
        Retrieves the batch source for user features.

        Returns:
            SourceDefinition: The batch source object for user features.
        """
        return SourceDefinition(self.raw_data_path).user_batch_source

    def _get_content_batch_source(self):
        """
        Retrieves the batch source for content features.

        Returns:
            SourceDefinition: The batch source object for content features.
        """
        return SourceDefinition(self.raw_data_path).content_info_batch_source

    def _get_ab_user_batch_source(self):
        """
        Retrieves the batch source for AB testing user features.

        Returns:
            SourceDefinition: The batch source object for AB testing user features.
        """
        return SourceDefinition(self.raw_data_path).ab_user_batch_source

    def _get_online_item_batch_source(self):
        """
        Retrieves the batch source for online item features.

        Returns:
            SourceDefinition: The batch source object for online item features.
        """
        return SourceDefinition(self.raw_data_path).online_item_batch_source

    def _get_online_user_batch_source(self):
        """
        Retrieves the batch source for online user features.

        Returns:
            SourceDefinition: The batch source object for online user features.
        """
        return SourceDefinition(self.raw_data_path).online_user_batch_source

    def _create_anchored_feature_list(self):
        """
        Creates a list of feature anchors that group data sources with
        their corresponding features.

        Feature anchors map batch sources to specific feature lists.

        Returns:
            list: A list of `FeatureAnchor` objects representing anchored
            feature groups.
        """
        user_features = [
            self.feature_transform.f_age_group,
            self.feature_transform.f_province,
            self.feature_transform.f_package_code,
            self.feature_transform.f_sex,
        ]
        user_feature_anchor = FeatureAnchor(
            name="userFeatures", source=self.user_batch_source, features=user_features
        )

        profile_features = [
            self.feature_transform.f_hashed_profile_id,
            self.feature_transform.f_hashed_profile_id_v2,
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
        """
        Creates a list of derived features based on transformations of existing features

        Derived features are defined using transformation logic provided by
        `FeatureTransform`.

        Returns:
            list: A list of derived feature definitions.
        """
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

    def build_features(self):
        """
        Builds features in the feature store by processing feature anchors and
        derived features.

        This method uses the Feathr client to process the `anchored_feature_list` and
        `derived_feature_list` and prepare them for usage in the feature store.
        """
        self.client.build_features(
            anchor_list=self.anchored_feature_list,
            derived_feature_list=self.derived_feature_list,
        )

    def register_features(self):
        """
        Registers features into the feature store.
        """
        try:
            self.client.register_features()
        except Exception as e:
            logger.warning(e)

    def run(self):
        """
        Executes the primary workflow of the FeatureRegistry.
        """
        self.build_features()
        # self.register_features()
