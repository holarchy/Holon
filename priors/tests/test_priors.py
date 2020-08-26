import pytest
import priors
import common
import tempfile
import json
import os

priors_tests_dir = os.path.dirname(os.path.abspath(__file__))
priors_dir = os.path.dirname(priors_tests_dir)
project_dir = os.path.dirname(priors_dir)

test_data_dir = os.path.join(project_dir, 'scripts', 'tests', 'test_data')
redacted_filename = os.path.join(test_data_dir, 'test_data_redacted.csv')
unredacted_filename = os.path.join(test_data_dir, 'test_data_unredacted.csv')

exceptions_filename = os.path.join(test_data_dir, 'test_data_exceptions.csv')


def test_priors_structure():
    with tempfile.TemporaryDirectory() as temp_dir:
        priors.make_priors(unredacted_filename, temp_dir, '19.43')

        # make sure priors file has the required structure
        inner_dirs = os.listdir(temp_dir)
        with open(os.path.join(temp_dir, inner_dirs[0], 'priors', '19.43', '.json')) as f:
            prior = json.load(f)

        for field in common.numeric_vars():
            p = prior[field]
            assert p.get('nan_length') is not None
            assert p.get('prior_length') is not None
            assert p.get('mean') is not None
            assert p.get('stdev') is not None
            assert p.get('cdf') is not None

        for field in common.binary_vars():
            p = prior[field]
            assert p.get('nan_length') is not None
            assert p.get('prior_length') is not None
            assert p.get('num_unique') is not None
            assert p.get('cdf') is not None
            assert p.get('binary_mean') is not None

        for field in common.categorical_vars():
            p = prior[field]
            assert p.get('nan_length') is not None
            assert p.get('prior_length') is not None
            assert p.get('num_unique') is not None
            assert p.get('cdf') is not None
            assert p.get('category_equivalent_distance') is not None
            assert p.get('stdev_scaler') is not None
            assert p.get('category_equivalent_stdev') is not None


@pytest.mark.xfail  # adding recursive priors, we no longer can easily force an exception in priors.
def test_priors_exceptions():

    with tempfile.TemporaryDirectory() as temp_dir:
        priors.make_priors(exceptions_filename, temp_dir, '19.43')

        # make sure priors file has the required structure
        inner_dirs = os.listdir(temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, inner_dirs[0], 'exceptions', '19.43.json'))
