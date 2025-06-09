# tests/test_utils.py
import unittest
import pandas as pd
import os
import sys

# Add the project root to the Python path to allow imports from src and rl
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils import load_market_data_from_db

# NOTE: This test suite assumes you have already run data_collector.py
# and have a populated market_data.db file in the data/ directory.

class TestUtils(unittest.TestCase):

    def test_load_market_data_successfully(self):
        """
        Tests if the function can successfully load data for valid tickers.
        """
        tickers = ['MSFT', 'GOOGL']
        features = ['close', 'rsi']
        df = load_market_data_from_db(
            tickers_list=tickers,
            feature_columns=features,
            start_date="2020-01-01",
            end_date="2020-03-31"
        )
        self.assertIsInstance(df, pd.DataFrame, "Function should return a pandas DataFrame.")
        self.assertFalse(df.empty, "DataFrame should not be empty for valid inputs.")
        
        # Check if expected columns are present
        expected_cols = ['MSFT_close', 'MSFT_rsi', 'GOOGL_close', 'GOOGL_rsi']
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Expected column '{col}' not in DataFrame.")

    def test_handle_invalid_ticker(self):
        """
        Tests if the function returns an empty DataFrame for a ticker that does not exist.
        """
        tickers = ['INVALIDTICKERXYZ']
        df = load_market_data_from_db(
            tickers_list=tickers,
            start_date="2020-01-01",
            end_date="2020-03-31"
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty, "DataFrame should be empty for invalid tickers.")

    def test_handle_invalid_feature(self):
        """
        Tests if the function handles requests for feature columns that don't exist in the DB.
        It should still return data for the valid columns.
        """
        tickers = ['AAPL']
        # 'invalid_feature' does not exist in our data_collector schema
        features = ['close', 'invalid_feature']
        df = load_market_data_from_db(
            tickers_list=tickers,
            feature_columns=features,
            start_date="2020-01-01",
            end_date="2020-03-31"
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "DataFrame should not be empty if at least one feature is valid.")
        self.assertIn('AAPL_close', df.columns)
        self.assertNotIn('AAPL_invalid_feature', df.columns)

    def test_date_range_filtering(self):
        """
        Tests if the data is correctly filtered by the provided start and end dates.
        """
        start_date = "2021-02-01"
        end_date = "2021-02-28"
        df = load_market_data_from_db(
            tickers_list=['SPY'],
            feature_columns=['close'],
            start_date=start_date,
            end_date=end_date
        )
        self.assertFalse(df.empty)
        # Check if all dates in the index are within the specified range
        self.assertTrue((df.index >= pd.to_datetime(start_date)).all())
        self.assertTrue((df.index <= pd.to_datetime(end_date)).all())


if __name__ == '__main__':
    unittest.main()