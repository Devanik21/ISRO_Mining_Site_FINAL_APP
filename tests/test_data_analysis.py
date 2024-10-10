import pytest
from data_analysis_module import preprocess_data  # Adjust based on your module structure

def test_preprocess_data():
    input_data = [None, 2, 3, None, 5]
    expected_output = [0, 2, 3, 0, 5]
    assert preprocess_data(input_data) == expected_output