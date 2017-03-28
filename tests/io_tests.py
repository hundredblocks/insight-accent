import numpy as np
import data_fetch
import os


def test_io():
    test_file_path = 'test_io'
    y = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    data_fetch.save_spectrogram_array(test_file_path, y)
    y_ = data_fetch.load_spectrogram_array(test_file_path)
    os.remove(test_file_path)
    assert all([og == rec for og, rec in zip(y, y_)])
