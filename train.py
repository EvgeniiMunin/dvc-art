import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from typing import Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from featurization import TrainingParams
from featurization import read_training_pipeline_params

SklearnRegressor = Union[RandomForestRegressor, LinearRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnRegressor:
    if train_params.model_type == "RandomForestRegressor":
        model = RandomForestRegressor(
            max_depth=train_params.max_depth, random_state=train_params.random_state
        )
    elif train_params.model_type == "LinearRegression":
        model = LogisticRegression(
            solver=train_params.solver,
            C=train_params.C,
            random_state=train_params.random_state,
        )
    model.fit(features, target)
    return model


def serialize_model(model: RandomForestRegressor, output: str) -> str:
    with open(output, "wb") as f:
        joblib.dump(model, f)
    return output


if __name__ == "__main__":
    config_path = "configs/train_config.yaml"
    training_pipeline_params = read_training_pipeline_params(config_path)

    train_features = pd.read_csv(training_pipeline_params.output_data_train_path)
    train_target = pd.read_csv(training_pipeline_params.output_target_train_path)

    model = train_model(
        train_features, train_target["price"], training_pipeline_params.train_params
    )

    serialize_model(model, training_pipeline_params.output_model_path)
