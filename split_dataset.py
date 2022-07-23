import argparse
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from featurization import read_training_pipeline_params, SplittingParams
import logging
import sys


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def split_train_val_data(
    data: pd.DataFrame, target: pd.Series, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_data, val_data, train_target, val_target = train_test_split(
        data, target, test_size=params.val_size, random_state=params.random_state
    )

    return train_data, val_data, train_target, val_target


if __name__ == "__main__":
    config_path = "configs/train_config.yaml"
    training_pipeline_params = read_training_pipeline_params(config_path)

    df = pd.read_csv(training_pipeline_params.output_data_featurized_path)
    target = pd.read_csv("data/data_target.csv")

    df_train, df_test, target_train, target_test = split_train_val_data(
        df, target, training_pipeline_params.splitting_params
    )

    logger.info((df_train.shape, df_test.shape))
    logger.info((target_train.shape, target_test.shape))

    df_train.to_csv(training_pipeline_params.output_data_train_path, index=False)
    df_test.to_csv(training_pipeline_params.output_data_test_path, index=False)

    target_train.to_csv(training_pipeline_params.output_target_train_path, index=False)
    target_test.to_csv(training_pipeline_params.output_target_test_path, index=False)
