import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions."""
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """Serializes model/encoder to a file."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """Loads pickle file from `path` and returns it."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model,
):
    """
    Computes model metrics on a slice of the data specified by column_name == slice_value.
    """
    slice_df = data[data[column_name] == slice_value]

    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
