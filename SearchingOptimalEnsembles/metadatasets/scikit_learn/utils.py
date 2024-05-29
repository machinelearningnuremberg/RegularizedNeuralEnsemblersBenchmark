from __future__ import annotations

import numpy as np


def calculate_errors(y_proba, y_labels):
    # Argmax over the last dimension to get class predictions
    y_predictions = np.argmax(y_proba, axis=-1)  # This results in shape (B, D, P)

    # We now need to compare these predictions with y_labels
    # Since y_labels is (B, D), we need to expand it to (B, D, P) for broadcasting
    y_labels_expanded = np.expand_dims(y_labels, axis=-1)  # Expanding the last dimension
    y_labels_expanded = np.repeat(
        y_labels_expanded, repeats=y_predictions.shape[-1], axis=-1
    )  # Repeat for each member

    # Compute accuracy for each ensemble member
    accuracy = (y_predictions == y_labels_expanded).mean(
        axis=1
    )  # Mean over D (datapoints)
    error = 1.0 - accuracy

    return error  # This returns the error for each member in each ensemble (shape B, P)
