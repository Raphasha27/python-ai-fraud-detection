from functools import lru_cache
from ..ml.model import FraudDetectionModel
from ..ml.features import FeatureEngineer

model_instance: Optional[FraudDetectionModel] = None
feature_engineer_instance: Optional[FeatureEngineer] = None


def get_model() -> FraudDetectionModel:
    global model_instance
    if model_instance is None:
        model_instance = FraudDetectionModel()
    return model_instance


def get_feature_engineer() -> FeatureEngineer:
    global feature_engineer_instance
    if feature_engineer_instance is None:
        feature_engineer_instance = FeatureEngineer()
    return feature_engineer_instance
