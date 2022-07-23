from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import logging
import sys
from typing import List
import yaml

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=42)


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: str = field(default="target")


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=42)
    # RF params
    max_depth: int = field(default=5)
    # LR params
    solver: str = field(default="lbfgs")
    C: float = field(default=1.0)


@dataclass()
class TrainingPipelineParams:
    output_data_featurized_path: str
    output_data_target_path: str
    output_data_train_path: str
    output_data_test_path: str
    output_target_train_path: str
    output_target_test_path: str
    output_model_path: str
    output_transformer_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    input_data_path: str = field(default="data/wines_SPA.csv")


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)
        schema = TrainingPipelineParamsSchema().load(config_dict)
        #logger.info(f"Check schema: {schema}")
        return schema


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)
