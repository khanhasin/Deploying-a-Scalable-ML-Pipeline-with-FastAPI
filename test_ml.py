import numpy as np
import pandas as pd

from ml.data import apply_label, process_data
from ml.model import compute_model_metrics, inference, train_model


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_apply_label_outputs_expected_strings():
    assert apply_label(np.array([1])) == ">50K"
    assert apply_label(np.array([0])) == "<=50K"


def test_compute_model_metrics_returns_valid_range():
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_train_model_and_inference_work_on_small_dataset():
    df = pd.read_csv("data/census.csv").head(200)
    X, y, encoder, lb = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    preds = inference(model, X[:10])
    assert preds.shape[0] == 10
    assert set(np.unique(preds)).issubset({0, 1})
    assert encoder is not None
    assert lb is not None
