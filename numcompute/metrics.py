import numpy as np


def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of a binary classification model given the true and
    predicted labels.

    Args:
        y_true (np.ndarray): Array of size (N,) containing true labels
        y_pred (np.ndarray): Array of size (N,) containing predicted labels

    Outputs:
        float: Accuracy of the model.
    
    Exceptions Raised:
        TypeError: If y_true is not of type np.ndarray
        TypeError: If y_pred is not of type np.ndarray
        ValueError: If y_true is not a 1D array
        ValueError: If y_pred is not a 1D array
        ValueError: If y_pred and y_true do not have same shape
        ValueError: If y_true contains values other than 0 or 1
        ValueError: If y_pred contains values other than 0 or 1
    
    Time Complexity: O(N)
    Space Complexity: O(N)
    """

    # Exception Handling
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a NumPy array.")
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be a 1D array.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.isin(y_true, [0, 1]).all() == False:
        raise ValueError("True labels must be either 0 or 1.")
    if np.isin(y_pred, [0, 1]).all() == False:
        raise ValueError("Predicted labels must be either 0 or 1.")

    # NaN Handling
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("Warning: NaNs were ignored to calculate accuracy.")
        return np.nanmean(y_true == y_pred)

    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    """
    Calculates the precision of a binary classification model given the true 
    and predicted labels.

    Args:
        y_true (np.ndarray): Array of size (N,) containing true labels
        y_pred (np.ndarray): Array of size (N,) containing predicted labels

    Outputs:
        float: Precision of the model.

    Exceptions Raised:
        TypeError: If y_true is not of type np.ndarray
        TypeError: If y_pred is not of type np.ndarray
        ValueError: If y_true is not a 1D array
        ValueError: If y_pred is not a 1D array
        ValueError: If y_pred and y_true do not have same shape
        ValueError: If y_true contains values other than 0 or 1
        ValueError: If y_pred contains values other than 0 or 1

    Time Complexity: O(N)
    Space Complexity: O(N)
    """

    # Exception Handling
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a NumPy array.")
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be a 1D array.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.isin(y_true, [0, 1]).all() == False:
        raise ValueError("True labels must be either 0 or 1.")
    if np.isin(y_pred, [0, 1]).all() == False:
        raise ValueError("Predicted labels must be either 0 or 1.")

    # NaN Handling
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("Warning: NaNs were ignored to calculate precision.")

    num_true_pos = np.where((y_true == 1) & (y_pred == 1))[0].size
    num_pred_pos = np.where(y_pred == 1)[0].size

    # If no predicted positive return 0 by convention
    if num_pred_pos == 0:
        return 0

    return num_true_pos / num_pred_pos


def recall(y_true, y_pred):
    """
    Calculates the recall of a binary classification model given the true and
    predicted labels.

    Args:
        y_true (np.ndarray): Array of size (N,) containing true labels
        y_pred (np.ndarray): Array of size (N,) containing predicted labels

    Outputs:
        float: Recall of the model.
    
    Exceptions Raised:
        TypeError: If y_true is not of type np.ndarray
        TypeError: If y_pred is not of type np.ndarray
        ValueError: If y_true is not a 1D array
        ValueError: If y_pred is not a 1D array
        ValueError: If y_pred and y_true do not have same shape
        ValueError: If y_true contains values other than 0 or 1
        ValueError: If y_pred contains values other than 0 or 1

    Time Complexity: O(N)
    Space Complexity: O(N)
    """

    # Exception Handling
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a NumPy array.")
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be a 1D array.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.isin(y_true, [0, 1]).all() == False:
        raise ValueError("True labels must be either 0 or 1.")
    if np.isin(y_pred, [0, 1]).all() == False:
        raise ValueError("Predicted labels must be either 0 or 1.")

    # NaN Handling
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("Warning: NaNs were ignored to calculate recall.")

    num_true_pos = np.where((y_true == 1) & (y_pred == 1))[0].size
    num_pos = np.where(y_true == 1)[0].size

    # If no actual positives return 0 by convention
    if num_pos == 0:
        return 0

    return num_true_pos / num_pos


def f1(y_true, y_pred):
    """
    Calculates the f1 score of a binary classification model given the true and
    predicted labels.

    Args:
        y_true (np.ndarray): Array of size (N,) containing true labels
        y_pred (np.ndarray): Array of size (N,) containing predicted labels

    Outputs:
        float: f1 score for the model
    
    Exceptions Raised:
        TypeError: If y_true is not of type np.ndarray
        TypeError: If y_pred is not of type np.ndarray
        ValueError: If y_true is not a 1D array
        ValueError: If y_pred is not a 1D array
        ValueError: If y_pred and y_true do not have same shape
        ValueError: If y_true contains values other than 0 or 1
        ValueError: If y_pred contains values other than 0 or 1

        (Exceptions are raised implicity through calling precision() and recall())
    
    Time Complexity: O(N)
    Space Complexity: O(N)
    """

    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    if prec + rec == 0:
        return 0

    return 2 * prec * rec / (prec + rec)


def confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix of a binary classification model given the true
    and predicted labels.

    Args:
        y_true (np.ndarray): Array of size (N,) containing true labels
        y_pred (np.ndarray): Array of size (N,) containing predicted labels

    Outputs:
        np.ndarray: Confusion matrix for the model
    
    Exceptions Raised:
        TypeError: If y_true is not of type np.ndarray
        TypeError: If y_pred is not of type np.ndarray
        ValueError: If y_true is not a 1D array
        ValueError: If y_pred is not a 1D array
        ValueError: If y_pred and y_true do not have same shape
        ValueError: If y_true contains values other than 0 or 1
        ValueError: If y_pred contains values other than 0 or 1
    
    Time Complexity: O(N)
    Space Complexity: O(N)
    """

    # Exception Handling
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a NumPy array.")
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be a 1D array.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.isin(y_true, [0, 1]).all() == False:
        raise ValueError("True labels must be either 0 or 1.")
    if np.isin(y_pred, [0, 1]).all() == False:
        raise ValueError("Predicted labels must be either 0 or 1.")

    # NaN Handling
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("Warning: NaNs were ignored to calculate confusion matrix.")

    num_true_pos = np.where((y_true == 1) & (y_pred == 1))[0].size
    num_false_pos = np.where((y_true == 0) & (y_pred == 1))[0].size
    num_false_neg = np.where((y_true == 1) & (y_pred == 0))[0].size
    num_true_neg = np.where((y_true == 0) & (y_pred == 0))[0].size

    conf_mat = np.array([[num_true_pos, num_false_pos],
                         [num_false_neg, num_true_neg]])

    return conf_mat


def mse(y_true, y_pred):
    """
    Calculates the MSE of a regression model given the true and
    predicted values.

    Args:
        y_true (np.ndarray): Array of size (N,) containing true outcome values
        y_pred (np.ndarray): Array of size (N,) containing predicted outcome
                             values

    Outputs:
        float: MSE of the regression model.
    
    Exceptions Raised:
        TypeError: If y_true is not of type np.ndarray
        TypeError: If y_pred is not of type np.ndarray
        ValueError: If y_true is not a 1D array
        ValueError: If y_pred is not a 1D array
        ValueError: If y_pred and y_true do not have same shape
    
    Time Complexity: O(N)
    Space Complexity: O(N)
    """

    # Exception Handling
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a NumPy array.")
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be a 1D array.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # NaN Handling
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("Warning: NaNs were ignored to calculate MSE.")
        return np.nanmean((y_true - y_pred) ** 2)

    return np.mean((y_true - y_pred) ** 2)
