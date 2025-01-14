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

PANDAS_DATE_FORMAT = "%Y-%m-%d"


class SpareFeatureInfo(metaclass=SingletonMeta):
    """
    A singleton class managing sparse feature encodings and hash configurations.

    This class defines and stores mappings for encoded and hashed sparse features, making it
    easier to process categorical data and configure hashing for feature engineering in large-scale
    systems. It supports efficient handling of feature indexing and hashing bucket sizes.

    Attributes:
        encoded_features (Dict[str, Dict[str, int]]): A dictionary mapping feature names to
            sub-dictionaries, where each sub-dictionary maps raw feature values to their
            corresponding encoded integer values.
        hashed_features (Dict[str, int]): A dictionary mapping hashed feature names (e.g.,
            user_id, item_id) to their predefined bucket sizes, used for hashing large feature spaces.
        num_user_to_bucket_size (Dict[int, int]): A dictionary mapping the number of users
            in the dataset to appropriate hash bucket sizes for user-related hashed features.
    """

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
    """
    An extended version of `SpareFeatureInfo` for managing double hashing configurations.

    This class inherits from `SpareFeatureInfo` and provides additional support for double
    hashing (dhash) features. It defines bucket sizes specific to dhash functionality and
    extends the hashing configurations for user and item features, accommodating higher
    scalability requirements.

    Attributes:
        num_user_to_dhash_bucket_size (Dict[int, int]): A mapping of the number of users
            in the dataset to suitable hash bucket sizes for double hashing (dhash) features.
        hashed_features (Dict[str, int]): A dictionary mapping dhash feature names
            (e.g., hashed_user_id_v2, hashed_item_id_v2) to their respective bucket sizes,
            as well as hashed features from the base class.
    """

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

ROLLING_PERIOD_FOR_USER_PREFER_TYPE = 30
ROLLING_PERIOD_FOR_POPULARITY_ITEM = 30

HASHED_CONTENT_CATEGORY_BS = 4001
HASHED_NUMBER_MONTHS_FROM_PUBLISH_BS = 997
HASHED_PUBLISH_YEAR_BIWEEKLY_BS = 3001
DATALOADER_BATCH_NUMBER = 41
NUMBER_OF_RANDOM_USER_GROUP = 10

INFERRING_USER_WEIGHT = 1
POSITIVE_WEIGH_FOR_POPULARITY_GROUP = [1, 1, 1.5, 2, 3]
DURATION_THRESHOLD_FOR_WEIGHTED_LR = 3600

clean_content_country_mapping = {
    # viet_nam
    "vietnam": "viet_nam",
    "việt_nam": "viet_nam",
    "1": "viet_nam",
    "vn": "viet_nam",
    # my
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

SELECTED_HISTORY_COLUMNS = [
    "content_id",
    "content_type",
    "username",
    "profile_id",
    "duration",
    "date_time",
    "filename_date",
    "part",
]

NUM_DATE_EVAL = 3
