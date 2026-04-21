import numpy as np

def load_csv(file_path, delimiter=",", missing_value=np.nan, skip_header=0):
    """
    Load CSV file into a NumPy array.

    Parameters:
        file_path (str): Path to CSV file
        delimiter (str): Delimiter used in file
        missing_value: Value to replace missing entries
        skip_header (int): Rows to skip at top

    Returns:
        np.ndarray: Loaded data

    Raises:
        ValueError: If file cannot be read
    """
    try:
        data = np.genfromtxt(
            file_path,
            delimiter=delimiter,
            skip_header=skip_header,
            filling_values=missing_value
        )
        return data
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")