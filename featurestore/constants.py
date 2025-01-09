"""
Module: constants

This module defines constants used throughout the codebase, including data type
configurations and standard data names. These constants help maintain consistency and
reduce hardcoded values across modules.
"""
import numpy as np

dtype = np.dtype


class DataName:
    """
    A class containing constant data names.
    """

    AB_TESTING_USER_INFO = "ab_testing_user_info"
    CONTENT_INFO = "content_info"
    CONTENT_TYPE = "content_type"
    MOVIE_HISTORY = "movie_watch_history"
    VOD_HISTORY = "vod_watch_history"
    ACCOUNT_MYTV_INFO = "user_mytv_info"
    PROFILE_MYTV_INFO = "profile_mytv_info"
    ONLINE_USER_FEATURES = "online_user_features"
    ONLINE_ITEM_FEATURES = "online_item_features"
    OBSERVATION_FEATURES = "observation_features"
    OFFLINE_FEATURES = "offline_features"
    USER_INFO = "user_info"
