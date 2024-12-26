from pathlib import Path
from typing import Dict

import numpy as np

from featurestore.base.utils.singleton import SingletonMeta

FILENAME_DATE_COL = "filename_date"
FILENAME_DATE_FORMAT = "%Y%m%d"
FILENAME_DATE_FORMAT_PYSPARK = "yyyyMMdd"

TIMESTAMP_COLUMN = "date_time"
TIMESTAMP_FORMAT = "yyyy-MM-dd"
ONLINE_INTERVAL = 90


class SpareFeatureInfo(metaclass=SingletonMeta):
    # encoded features with key is feature name and value is dict of encoded value
    # feature_name: {encoded_value: index}
    encoded_features: Dict[str, Dict] = {
        # should start from 1, 0 is for unknown.
        "encoded_content_country": {
            "trung_quoc": 1,
            "viet_nam": 2,
            "han_quoc": 3,
            "my": 4,
            "empty": 5,
            "nhat_ban": 6,
            "hong_kong": 7,
            "thai_lan": 8,
            "dai_loan": 9,
            "an_do": 10,
            "other": 11,
            "singapore": 12,
            "anh": 13,
            "duc": 14,
            "nga": 15,
            "uc": 16,
            "tay_ban_nha": 17,
            "colombia": 18,
            "canada": 19,
            "phap": 20,
            "tho_nhi_ky": 21,
            "philippines": 22,
            "bo_dao_nha": 23,
            "iceland": 24,
            "mexico": 25,
            "y": 26,
            "indonesia": 27,
            "nam_phi": 28,
            "serbia": 29,
            "israel": 30,
            "mong_co": 31,
            "kazakhstan": 32,
            "ireland": 33,
            "dan_mach": 34,
            "ucraina": 35,
        },
        "encoded_content_parent_type": {
            "tv_series": 1,
            "vod": 2,
            "movie": 3,
            "kid": 4,
            "clip": 5,
            "sport": 6,
            "empty": 7,
        },
        "encoded_content_type": {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "11": 11,
            "12": 12,
            "13": 13,
            "14": 14,
            "15": 15,
            "17": 16,
            "18": 17,
            "19": 18,
            "20": 19,
            "21": 20,
            "22": 21,
            "23": 22,
            "24": 23,
            "25": 24,
            "27": 25,
            "29": 26,
            "34": 27,
            "35": 28,
            "36": 29,
            "37": 30,
            "39": 31,
            "41": 32,
            "43": 33,
        },
        "encoded_user_province": {
            "HCM": 1,
            "THA": 2,
            "DLK": 3,
            "HPG": 4,
            "HNI": 5,
            "DNI": 6,
            "QNH": 7,
            "NDH": 8,
            "HDG": 9,
            "KHA": 10,
            "LAN": 11,
            "DNG": 12,
            "BDH": 13,
            "LDG": 14,
            "BTN": 15,
            "HNM": 16,
            "KGG": 17,
            "QTI": 18,
            "HBH": 19,
            "QBH": 20,
            "AGG": 21,
            "GLI": 22,
            "BGG": 23,
            "VTU": 24,
            "BDG": 25,
            "PTO": 26,
            "TGG": 27,
            "CTO": 28,
            "TBH": 29,
            "TNH": 30,
            "NBH": 31,
            "BTE": 32,
            "TNN": 33,
            "CMU": 34,
            "HYN": 35,
            "BPC": 36,
            "LSN": 37,
            "CBG": 38,
            "LCI": 39,
            "HUE": 40,
            "SLA": 41,
            "BLU": 42,
            "TQG": 43,
            "BNH": 44,
            "VPC": 45,
            "PYN": 46,
            "YBI": 47,
            "STG": 48,
            "HTH": 49,
            "QNI": 50,
            "QNM": 51,
            "KTM": 52,
            "DTP": 53,
            "VLG": 54,
            "DKG": 55,
            "TVH": 56,
            "NAN": 57,
            "NTN": 58,
            "HGG": 59,
            "BCN": 60,
            "LCU": 61,
            "DBN": 62,
            "HGI": 63,
            "empty": 64,
        },
        "encoded_user_package_code": {
            "MYTV006": 1,
            "MYTV021": 2,
            "MYTV008": 3,
            "MYTV010": 4,
            "MYTV014": 5,
            "MYTV015": 6,
            "MYTV013": 7,
            "MYTV016": 8,
            "MYTV012": 9,
            "FLX001": 10,
            "FLX003": 11,
            "VASC006": 12,
            "VASC005": 13,
            "VASC007": 14,
            "VASC011": 15,
            "VASC000": 16,
            "MYTV020": 17,
            "MYTV019": 18,
            "empty": 19,
        },
        "encoded_age_group": {
            "child": 1,
            "student": 2,
            "play": 3,
            "married": 4,
            "senior": 5,
            "older": 6,
        },
    }

    def get_number_of_embedding_inputs(self) -> Dict[str, int]:
        """Get number of embedding inputs. Hashed features will be the hash bucket size.
        Encoded features will be the largest value."""

        def _get_encoded_feature_size(feature_name: str) -> int:
            return max(self.encoded_features[feature_name].values()) + 1

        num_inputs = {
            feature: _get_encoded_feature_size(feature)
            for feature in self.encoded_features.keys()
        }
        num_inputs.update(self.hashed_features)
        return num_inputs

    num_user_to_bucket_size = {
        10000: 100003,
        100000: 1000003,
        200000: 2000003,
        500000: 4999999,
        1000000: 9999991,
    }

    def _closest_upper(self, myArr, myNumber):
        myArr = np.array(myArr)
        return myArr[myArr >= myNumber].min()

    def __init__(self, num_user=100000, update=False):
        bucket_size = self.num_user_to_bucket_size[
            self._closest_upper(list(self.num_user_to_bucket_size.keys()), num_user)
        ]
        self.hashed_features: Dict[str, int] = {
            "hashed_user_id": bucket_size,
            "hashed_item_id": bucket_size,
            "hashed_content_category": 4001,
            "hashed_num_months_from_publish": 997,
            "hashed_publish_year_biweekly": 3001,
        }


class DhashSpareFeatureInfo(SpareFeatureInfo):
    num_user_to_dhash_bucket_size = {
        10000: 10007,
        100000: 100003,
        200000: 199999,
        300000: 300007,
        500000: 500009,
        1000000: 1000003,
        2000000: 2000003,
        5000000: 4999999,
        10000000: 9999991,
        20000000: 20000000,
        35000000: 35000000,
        50000000: 49999991,
    }

    def __init__(self, num_user=100000, update=False):
        bucket_size = self.num_user_to_dhash_bucket_size[
            self._closest_upper(
                list(self.num_user_to_dhash_bucket_size.keys()), num_user
            )
        ]
        self.hashed_features: Dict[str, int] = {
            "hashed_user_id": bucket_size,
            "hashed_user_id_v2": bucket_size,
            "hashed_item_id": bucket_size,
            "hashed_item_id_v2": bucket_size,
            "hashed_content_category": 4001,
            "hashed_num_months_from_publish": 997,
            "hashed_publish_year_biweekly": 3001,
        }


VOD_DIRTY_CLICK_DURATION_THRESHOLD = 60
MOVIE_DIRTY_CLICK_DURATION_THRESHOLD = 120

CHANNEL_TYPE_GROUP = ["1", "17", "18", "23", "24"]
MOVIE_TYPE_GROUP = ["2", "20", "21", "22", "25", "29"]
VOD_TYPE_GROUP = [
    "0",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "19",
    "27",
    "34",
    "35",
    "36",
    "37",
    "39",
    "41",
    "43",
    "47",
    "78",
]

ROLLING_PERIOD_FOR_PREFER_TYPE = 30

HASHED_CONTENT_CATEGORY_BS = 4001
HASHED_NUMBER_MONTHS_FROM_PUBLISH_BS = 997
HASHED_PUBLISH_YEAR_BIWEEKLY_BS = 3001
DATALOADER_BATCH_NUMBER = 41
NUMBER_OF_RANDOM_USER_GROUP = 10

INFERRING_USER_WEIGHT = 1
POSITIVE_WEIGH_FOR_POPULARITY_GROUP = [1, 1, 1.5, 2, 3]
DURATION_THRESHOLD_FOR_WEIGHTED_LR = 3600
