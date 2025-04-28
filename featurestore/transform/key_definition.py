"""
Module: key_definition

This module defines the `KeyDefinition` singleton class, which is responsible for
managing and standardizing key metadata used in feature definitions and transformations.
The centralized management of keys ensures consistency and reusability across feature
pipelines and the feature store.
"""
from feathr import TypedKey, ValueType

from featurestore.base.utils.singleton import SingletonMeta


class KeyDefinition(metaclass=SingletonMeta):
    """
    KeyDefinition is a singleton class responsible for defining
    and managing key metadata for feature definitions.

    The class provides standardized key definitions for use in feature transformations.
    It centralizes key metadata to ensure consistency across the feature store.

    Attributes:
        user_key (TypedKey): A typed key for user-related features,
        containing metadata like column name and type.
        item_key (TypedKey): A typed key for item-related features,
        containing metadata like column name and type.
        key_collection (dict): A dictionary mapping key names to
        their corresponding `TypedKey` objects.
    """

    def __init__(
        self,
    ):
        self.profile_key = TypedKey(
            key_column="profile_id",
            key_column_type=ValueType.STRING,
            description="Profile id",
            full_name="mytv.profile_id",
        )
        self.item_key = TypedKey(
            key_column="item_id",
            key_column_type=ValueType.STRING,
            description="item id",
            full_name="mytv.item_id",
        )

        self.key_collection = {
            "profile_id": self.profile_key,
            "item_id": self.item_key,
        }
