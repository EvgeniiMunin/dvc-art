import json
import pandas as pd
from sklearn.metrics import r2_score
import joblib

from featurization import read_training_pipeline_params


if __name__ == "__main__":
    config_path = "configs/train_config.yaml"
    training_pipeline_params = read_training_pipeline_params(config_path)

    test_features = pd.read_csv(training_pipeline_params.output_data_test_path)
    test_target = pd.read_csv(training_pipeline_params.output_target_test_path)

    model = joblib.load(training_pipeline_params.output_model_path)

    predicts = model.predict(test_features)
    r2_score(predicts, test_target)

    json.dump(
        obj={"r2_score": r2_score(predicts, test_target)},
        fp=open(training_pipeline_params.metric_path, "w"),
    )
