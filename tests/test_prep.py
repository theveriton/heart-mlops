import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import data_prep


def test_load_data():
    df = data_prep.load_data()
    assert df is not None
    assert "target" in df.columns
    assert len(df) > 100