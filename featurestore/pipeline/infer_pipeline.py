import pandas as pd

from featurestore.base.schemas.pipeline_config import InferPipelineConfig
from featurestore.base.utils.config import parse_infer_config
from featurestore.base.utils.utils import return_or_load


class InferPipeline:
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
        infer_features = self.client.multi_get_online_features(
            feature_table=feature_table_name,
            keys=key_list,
            feature_names=feature_name_list,
        )
        infer_features_df = pd.DataFrame(infer_features).T.reset_index()
        infer_features_df.columns = key_names + feature_name_list
        return infer_features_df

    def _join_infer_data(self):
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

    def _postprocess_data(self, df):
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
        joined_df = self._join_infer_data()
        result_df = self._postprocess_data(joined_df)
        return result_df
