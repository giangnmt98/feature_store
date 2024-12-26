from feathr import TypedKey, ValueType

from featurestore.base.utils.singleton import SingletonMeta


class KeyDefinition(metaclass=SingletonMeta):
    def __init__(
        self,
    ):
        self.user_key = TypedKey(
            key_column="user_id",
            key_column_type=ValueType.STRING,
            description="user id",
            full_name="mytv.user_id",
        )
        self.item_key = TypedKey(
            key_column="item_id",
            key_column_type=ValueType.STRING,
            description="item id",
            full_name="mytv.item_id",
        )

        self.key_collection = {
            "user_id": self.user_key,
            "item_id": self.item_key,
        }
