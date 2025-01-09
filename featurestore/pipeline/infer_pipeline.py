"""
Module: infer_pipeline

This module defines the `InferPipeline` class, which is responsible for enriching
user-item interaction data with features retrieved from online feature tables.
The pipeline handles feature retrieval, data merging, and postprocessing to ensure
the resulting dataset is suitable for downstream tasks such as recommendations or
predictions.
"""
import pandas as pd

from featurestore.base.schemas.pipeline_config import InferPipelineConfig
from featurestore.base.utils.config import parse_infer_config
from featurestore.base.utils.utils import return_or_load


class InferPipeline:
    """
    InferPipeline class handles the process of enriching a user-item interaction dataset
    with features retrieved from online feature tables. It also performs postprocessing
    to deal with missing or new user data.

    Attributes:
        user_item_df (pd.DataFrame): The dataframe containing user and item
            interaction data.
        infer_config (InferPipelineConfig): Configuration object parsed from the given
            config file.
        client (FeathrClient): Client used to interact with the feature store.
        user_id_list (ndarray): Unique list of user IDs from the input dataframe.
        item_id_list (ndarray): Unique list of item IDs from the input dataframe.
        user_key (str): Column name representing user keys, defaulted to "user_id".
        item_key (str): Column name representing item keys, defaulted to "item_id".
    """

    def __init__(
        self,
        feathr_client,
        user_item_df,
        config_path: str,
    ):
        self.user_item_df = user_item_df
        assert (
            not self.user_item_df.empty
        ), "Input dataframe must not be empty. Stop getting online feature."
        self.infer_config = return_or_load(
            config_path, InferPipelineConfig, parse_infer_config
        )
        self.client = feathr_client
        self.user_id_list = self.user_item_df["user_id"].unique()
        self.item_id_list = self.user_item_df["item_id"].unique()
        self.user_key = "user_id"
        self.item_key = "item_id"

    def _get_online_features(
        self, feature_table_name, key_list, feature_name_list, key_names
    ):
        """
        Retrieves online features from the specified feature table for the given keys
        and feature names.

        Args:
            feature_table_name (str): Name of the feature table from which features
                will be retrieved.
            key_list (list): List of keys for which features will be fetched.
            feature_name_list (list): List of feature names to be retrieved from the
                feature table.
            key_names (list): List of column names corresponding to the provided keys.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the retrieved features, where
                the columns include the key columns and the specified feature names.
        """
        infer_features = self.client.multi_get_online_features(
            feature_table=feature_table_name,
            keys=key_list,
            feature_names=feature_name_list,
        )
        infer_features_df = pd.DataFrame(infer_features).T.reset_index()
        infer_features_df.columns = key_names + feature_name_list
        return infer_features_df

    def join_infer_data(self):
        """
        Joins the input user-item dataframe with features retrieved from multiple
        online feature tables.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the original user-item data
            along with the features fetched from the online feature tables.
        """
        joined_df = self.user_item_df.copy()
        for table in self.infer_config.feature_tables:
            infer_feature_df = self._get_online_features(
                feature_table_name=table.feature_table_names[0],
                key_list=self.user_id_list
                if self.user_key in table.key_names
                else self.item_id_list,
                feature_name_list=table.feature_names,
                key_names=table.key_names,
            )
            joined_df = joined_df.merge(
                infer_feature_df, on=table.key_names, how="left"
            )
        return joined_df

    def postprocess_data(self, df):
        """
        Applies postprocessing to the given dataframe to handle missing values
        and new users.

        Args:
            df (pd.DataFrame): The dataframe resulting from joining the user-item
                data with the retrieved features. It may contain missing values.

        Returns:
            pd.DataFrame: A pandas DataFrame with missing values filled and new users
                appropriately handled.
        """
        # get fill nan value from online feature store
        feature_table_list = self.infer_config.feature_tables
        user_table = [i for i in feature_table_list if self.user_key in i.key_names][0]
        empty_user_df = self._get_online_features(
            feature_table_name=user_table.feature_table_names[0],
            key_list=["0#empty"],
            feature_name_list=user_table.feature_names,
            key_names=user_table.key_names,
        )
        fillna_value_dict = empty_user_df.to_dict(orient="records")[0]

        # get new users
        finding_cols = user_table.feature_names
        new_user_list = df[df[finding_cols].isna().all(axis=1)][self.user_key].tolist()

        # fill nan value
        new_user_df = df[df[self.user_key].isin(new_user_list)]
        new_user_df = new_user_df.fillna(value=fillna_value_dict)

        old_user_df = df[~df[self.user_key].isin(new_user_list)]
        df = pd.concat([old_user_df, new_user_df])
        return df

    def run(self):
        """
        Runs the entire inference pipeline to enrich the user-item interaction data
        with features and apply postprocessing.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the enriched user-item data
            with all necessary features and postprocessing applied.
        """

        joined_df = self.join_infer_data()
        result_df = self.postprocess_data(joined_df)
        return result_df
