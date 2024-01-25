import numpy as np
from scipy.stats import pearsonr

def pearson_correlation(y_true, y_pred):
    """
    Calculate the Pearson correlation coefficient between true and predicted images.

    Args:
        y_true (ndarray): True images.
        y_pred (ndarray): Predicted images.

    Returns:
        float: Pearson correlation coefficient.
    """
    return pearsonr(y_true.flatten(), y_pred.flatten())[0]

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted images.

    Args:
        y_true (ndarray): True images.
        y_pred (ndarray): Predicted images.

    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
