import unittest
from unittest.mock import patch, mock_open, MagicMock
import string
from ..src.ingest_data import (
    fetch_housing_data,
    load_housing_data
)

class IngestDataTest(unittest.TestCase):
    def test_fetch_housing_data(self, mock_urlretrieve, mock_makedirs) -> None:
        housing_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
        housing_path = os.path.join("datasets", "housing")
        fetch_housing_data(housing_url, housing_path)
        
        # Assert that functions were called with correct arguments
        mock_makedirs.assert_called_once_with(housing_path, exist_ok=True)
        mock_urlretrieve.assert_called_once_with(housing_url)
        
        with self.assertRaises(Exception) as context:
            fetch_housing_data(housing_url, housing_path)

        self.assertEqual(str(context.exception), 'Failed to create directory')
    
    def test_load_housing_data(self, mock_read_csv):
        housing_path = os.path.join("datasets", "housing")
        result = load_housing_data(housing_path)
        
        self.assertEqual(result.equals(pd.DataFrame()), True)  # Check if empty DataFrame is returned


if __name__ == '__main__':
    unittest.main()
        