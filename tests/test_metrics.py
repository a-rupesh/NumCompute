import numpy as np
from metrics import accuracy, precision, recall, f1, confusion_matrix, mse

def accuracy_test():
    """
    Testing for accuracy() function.
    """

    # Test 1: Perfect prediction
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    assert accuracy(y_true, y_pred) == 1.0

    # Test 2: Half correct
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert accuracy(y_true, y_pred) == 0.5

    # Test 3: All wrong
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    assert accuracy(y_true, y_pred) == 0.0

    print("accuracy() passed all tests.")

def precision_test():
    """
    Testing for precision() function.
    """

    # Test 1: No false positives
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    assert precision(y_true, y_pred) == 1.0

    # Test 2: Some false positives
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 0])  # FP = 1
    tp = 2
    fp = 1
    assert precision(y_true, y_pred) == tp / (tp + fp)

    # Tes 3: No predicted positives
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert precision(y_true, y_pred) == 0.0

    print("precision() passed all tests.")

def recall_test():
    """
    Testing for recall() function.
    """

    # Test 1: Perfect recall (no false negatives)
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    assert recall(y_true, y_pred) == 1.0

    # Test 2: Some false negatives
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])  # FN = 1
    tp = 1
    fn = 1
    assert recall(y_true, y_pred) == tp / (tp + fn)

    # Test 3: No actual positives
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert recall(y_true, y_pred) == 0.0

    print("recall() passed all tests.")

def f1_test():
    """
    Testing for f1() function.
    """

    # Test 1: Perfect prediction
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    assert f1(y_true, y_pred) == 1

    # Test 2: Mixed case
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    tp = 1
    fp = 1
    fn = 1
    precision_val = tp / (tp + fp)
    recall_val = tp / (tp + fn)
    f1_expected = 2 * precision_val * recall_val / (precision_val + recall_val)
    assert f1(y_true, y_pred) == f1_expected

    # Test 3: No predicted positives
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert f1(y_true, y_pred) == 0

    print("f1() passed all tests.")

def confusion_matrix_test():
    """
    Testing for confusion_matrix() function.
    """

    # Test 1: Perfect prediction
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    assert (confusion_matrix(y_true, y_pred) == np.array([[2, 0], [0, 2]])).all()

    # Test 2: Mixed case
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert (confusion_matrix(y_true, y_pred) == np.array([[1, 1], [1, 1]])).all()

    # Test 3: All negatives predicted
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert (confusion_matrix(y_true, y_pred) == np.array([[0, 0], [2, 2]])).all()

    print("confusion_matrix() passed all tests.")

def mse_test():
    """
    Testing for mse() function.
    """

    # Test 1: Perfect prediction
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    assert mse(y_true, y_pred) == 0

    # Test 2: Simple case
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 2])
    assert np.isclose(mse(y_true, y_pred), 2/3)

    # Test 3: Mixed positive/negative values
    y_true = np.array([-1, 0, 1])
    y_pred = np.array([1, 0, -1])
    assert np.isclose(mse(y_true, y_pred), 8/3)

    # Test 4: Compare directly to NumPy function
    y_true = np.array([3.5, 2.0, -1.0, 4.0])
    y_pred = np.array([3.0, 2.5, -2.0, 5.0])
    assert np.isclose(mse(y_true, y_pred),
                      np.mean((y_true - y_pred) ** 2))

    # Test 5: All zeros
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    assert mse(y_true, y_pred) == 0.0

    print("mse() passed all tests.")

# Run tests
accuracy_test()
precision_test()
recall_test()
f1_test()
confusion_matrix_test()

print("\nAll tests were passed.")
