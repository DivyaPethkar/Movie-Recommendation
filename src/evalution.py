import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(predictions, true_ratings):
    """
    Evaluates model predictions using common metrics.

    Args:
        predictions (list): List of predicted ratings.
        true_ratings (list): List of true ratings.

    Returns:
        dict: Dictionary of evaluation metrics.
    """

    rmse = np.sqrt(mean_squared_error(predictions, true_ratings))
    mae = mean_absolute_error(predictions, true_ratings)

    return {
        "RMSE": rmse,
        "MAE": mae
    }
