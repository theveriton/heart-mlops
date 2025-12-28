import pandas as pd

def test_data_exists():
    df = pd.read_csv("data/raw/heart.csv")
    assert len(df) > 100
    