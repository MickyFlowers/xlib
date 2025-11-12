import collections
import numpy as np

class MovingAverageFilter:
    """
    A simple moving average filter class.
    """
    def __init__(self, window_size):
        """
        Initializes the moving average filter.
        :param window_size: The size of the data window for averaging.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        
        self.window_size = window_size
        self.data_window = collections.deque(maxlen=window_size)

    def update(self, new_value):
        """
        Adds a new value to the filter and returns the updated moving average.
        Handles both single float values and numpy arrays.
        """
        self.data_window.append(new_value)
        return np.mean(self.data_window, axis=0)

    def __call__(self, new_value):
        """
        Allows the filter object to be called like a function.
        """
        return self.update(new_value)

    def clear(self):
        """
        Resets the filter's data window.
        """
        self.data_window.clear()

    @property
    def data(self):
        """
        Returns the current average without adding a new value.
        Returns 0 (or a zero vector) if the window is empty.
        """
        if not self.data_window:
            # If the first item added was an array, we should return a zero vector of the same shape.
            # For simplicity, we return 0, but this could be made more robust if needed.
            return None
        return np.mean(self.data_window, axis=0)
