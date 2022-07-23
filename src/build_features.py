import sys
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.params import FeatureParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def build_numerical_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler())])


def build_categorical_pipeline() -> Pipeline:
    return Pipeline([("ohe", OneHotEncoder())])


def process_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    df.fillna("nan", inplace=True)
    return pd.DataFrame(categorical_pipeline.fit_transform(df).toarray())


def process_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    numerical_pipeline = build_numerical_pipeline()
    df.fillna(0, inplace=True)
    return pd.DataFrame(numerical_pipeline.fit_transform(df))


def process_features(
    categorical_pipeline: Pipeline,
    numerical_pipeline: Pipeline,
    df: pd.DataFrame,
    params: FeatureParams,
) -> pd.DataFrame:
    categorical_df = pd.DataFrame(
        categorical_pipeline.fit_transform(df[params.categorical_features]).toarray()
    )
    numerical_df = pd.DataFrame(
        numerical_pipeline.fit_transform(df[params.numerical_features])
    )
    return pd.concat([numerical_df, categorical_df], axis=1)


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]
