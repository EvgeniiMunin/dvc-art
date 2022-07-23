import pandas as pd

from src.build_features import (
    extract_target,
    process_features,
    build_categorical_pipeline,
    build_numerical_pipeline,
)
from src.params import *


def process_features_targets(
    params: TrainingPipelineParams,
) -> (pd.DataFrame, pd.DataFrame):
    df = pd.read_csv(params.input_data_path)
    df.dropna(inplace=True)

    categorical_pipeline = build_categorical_pipeline()
    numerical_pipeline = build_numerical_pipeline()

    train_features = process_features(
        categorical_pipeline, numerical_pipeline, df, params.feature_params
    )
    train_target = extract_target(df, params.feature_params)

    return train_features, train_target


if __name__ == "__main__":
    config_path = "configs/train_config.yaml"

    training_pipeline_params = read_training_pipeline_params(config_path)

    features, target = process_features_targets(training_pipeline_params)

    logger.info(features.head())
    logger.info(target.head())
    features.to_csv(training_pipeline_params.output_data_featurized_path, index=False)
    target.to_csv(training_pipeline_params.output_data_target_path, index=False)
