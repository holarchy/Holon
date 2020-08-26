import common
import pytest
import os
import pandas as pd

common_tests_dir = os.path.dirname(os.path.abspath(__file__))
outliers_dir = os.path.dirname(common_tests_dir)
project_dir = os.path.dirname(outliers_dir)

test_data_dir = os.path.join(project_dir, 'scripts', 'tests', 'test_data')


def test_compare_versions():

    assert common.compare_versions('1.1.0', '1.1.1') is True
    assert common.compare_versions('1.2.0', '1.1.1', 'minor') is False
    assert common.compare_versions('1.2.0', '1.2.1', 'minor') is True
    assert common.compare_versions('1.2.3', '11.1.0', 'major') is False
    assert common.compare_versions('100.2.0', '100.1.0', 'major') is True
    assert common.compare_versions('1.2.0', '1.2.1', 'revision') is False
    assert common.compare_versions('1.2.0', '1.2.0', 'revision') is True

    with pytest.raises(Exception):
        common.compare_versions('1.2.0', '1.1.0', 'invalid_phrase')


def test_file_headers():
    # Test files with additional headers or missing data. No asserts for now; just sending user feedback through CLI
    df = pd.read_csv(os.path.join(test_data_dir, 'test_data_duplicate_headers.csv'))
    common.validate_headers(df, common.FILE_CONFIG.biflow_fields)

    df = pd.read_csv(os.path.join(test_data_dir, 'test_data_incorrect_fields.csv'))
    common.validate_headers(df, common.FILE_CONFIG.biflow_fields)

    df = pd.read_csv(os.path.join(test_data_dir, 'test_data_missing_fields.csv'))
    common.validate_headers(df, common.FILE_CONFIG.biflow_fields)
