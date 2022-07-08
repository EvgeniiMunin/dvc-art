import pandas as pd
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml
import logging
import sys
from typing import List, Optional

from build_features import build_transformer, extract_target, process_features, build_categorical_pipeline, build_numerical_pipeline
from params import *


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)
        schema = TrainingPipelineParamsSchema().load(config_dict)
        logger.info(f"Check schema: {schema}")
        return schema


def process_features_targets(params: TrainingPipelineParams) -> (pd.DataFrame, pd.DataFrame):
    df = pd.read_csv(params.input_data_path)
    df.dropna(inplace=True)

    transformer = build_transformer(params.feature_params)
    transformer.fit(df)

    categorical_pipeline = build_categorical_pipeline()
    numerical_pipeline = build_numerical_pipeline()

    train_features = process_features(
        categorical_pipeline,
        numerical_pipeline,
        df,
        params.feature_params
    )
    train_target = extract_target(df, params.feature_params)

    return train_features, train_target

if __name__ == "__main__":
    config_path = "configs/train_config.yaml"

    training_pipeline_params = read_training_pipeline_params(config_path)

    features, target = process_features_targets(
        training_pipeline_params
    )

    logger.info(features.head())
    logger.info(target.head())
    features.to_csv('data/data_featurized.csv', index=False)
    target.to_csv('data/data_target.csv', index=False)

