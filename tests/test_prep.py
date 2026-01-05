from src import data_prep


def test_load_data():
    df = data_prep.load_data()
    assert df is not None
    assert "target" in df.columns
    assert len(df) > 100