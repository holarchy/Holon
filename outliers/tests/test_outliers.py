import pytest
from outliers import outliers, predict_outliers
import os
import common
import json
import tempfile

outliers_tests_dir = os.path.dirname(os.path.abspath(__file__))
outliers_dir = os.path.dirname(outliers_tests_dir)
project_dir = os.path.dirname(outliers_dir)

test_data_dir = os.path.join(project_dir, 'scripts', 'tests', 'test_data')
redacted_filename = os.path.join(test_data_dir, 'test_data_redacted.csv')
unredacted_filename = os.path.join(test_data_dir, 'test_data_unredacted.csv')
redacted_priors_dir = os.path.join(test_data_dir, 'test_priors_redacted')
unredacted_priors_dir = os.path.join(test_data_dir, 'test_priors_unredacted')


def test_outliers_structure():

    for t in ['redacted', 'unredacted']:

        if t == 'redacted':
            filename = redacted_filename
            priors_dir = redacted_priors_dir
        else:
            filename = unredacted_filename
            priors_dir = unredacted_priors_dir

        with tempfile.TemporaryDirectory() as temp_dir:
            predict_outliers(filename, temp_dir, '19.43', priors_dir)

            # make sure there are no exceptions
            exceptions_dir = os.listdir(os.path.join(temp_dir, os.listdir(temp_dir)[0], 'exceptions'))
            assert len(exceptions_dir) == 0

            # make sure priors file has the required structure
            inner_dirs = os.listdir(temp_dir)
            with open(os.path.join(temp_dir, inner_dirs[0], 'metadata.json')) as f:
                metadata = json.load(f)

            for field in ['md5', 'filename', 'size (GB)', 'number of rows', 'subnet used', 'priors_directory',
                          'start date', 'end date', 'package version', 'outlier prediction date']:
                metadata[field]  # make sure field exists. if not, raise error

            assert metadata['package version'] == '0.2.1'  # hard-code in unit test to update after breaking change.

            predictions_dir = os.path.join(temp_dir, inner_dirs[0], 'predictions')
            prediction_files = os.listdir(predictions_dir)
            assert len(prediction_files) == 2

            with open(os.path.join(predictions_dir, prediction_files[0])) as f:
                outlier = json.load(f)

            assert 0 <= outlier.get('prediction') <= 1
            assert type(outlier.get('raw_data')) == dict
            assert outlier['objects'][0]['subjects'][0]['value'] is not None
            # for prediction in outlier['machine_predictions'][0]['tier_predictions'][0]['field_predictions']:
            #     assert prediction['field'] in common.FILE_CONFIG.uniflow_fields.keys()

            assert type(outlier) == dict

            # assert type(outlier.get('machine_predictions')) == list
            # for p in outlier.get('machine_predictions'):
            #     assert 0 <= p.get('prediction') <= 1
            #     assert p.get('ip') is not None
            #     assert p.get('subnet') is not None
            #     assert type(p.get('tier_predictions')) == list
            #     for q in p.get('tier_predictions'):
            #         assert 0 <= q.get('prediction') <= 1
            #         assert q.get('tier') is not None
            #         assert type(q.get('field_predictions')) == list
            #         for d in q.get('field_predictions'):
            #             assert 0 <= d.get('prediction') <= 1
            #             assert d.get('field') is not None
            #             assert 0 <= d.get('score') <= 1
            #             assert d.get('score_weight') > 0
            #             assert type(d.get('score_components')) == dict
            #             for c, v in d.get('score_components').items():
            #                 assert 0 <= v.get('value') <= 1
            #                 assert v.get('scaler') > 0
            #             assert 0 <= d.get('confidence') <= 1
            #             assert d.get('confidence_weight') > 0
            #             assert type(d.get('confidence_components')) == dict
            #             for c, v in d.get('confidence_components').items():
            #                 assert 0 <= v.get('value') <= 1
            #                 assert v.get('scaler') > 0


def test_confidence_functions():

    # outliers.conf_num_records()
    assert 0 <= outliers.conf_num_records(1000, scaler=0.5)['value'] <= 1
    with pytest.raises(ValueError):
        outliers.conf_num_records(1000, scaler=-0.5)

    # outliers.conf_binary_mean()
    assert 0 <= outliers.conf_binary_mean({'true': 100}, scaler=0.5)['value'] <= 1
    with pytest.raises(ValueError):
        outliers.conf_binary_mean({'true': 100}, scaler=-0.5)

    # outliers.conf_coeff_of_variation()
    assert 0 <= outliers.conf_coeff_of_variation(stdev=1, mean=10, scaler=0.5)['value'] <= 1
    with pytest.raises(ValueError):
        outliers.conf_coeff_of_variation(stdev=1, mean=10, scaler=-0.5)

    # outliers.conf_category_equivalent_stdev()
    assert 0 <= outliers.conf_category_equivalent_stdev(category_equivalent_stdev=50, scaler=0.5)['value'] <= 1
    with pytest.raises(ValueError):
        outliers.conf_category_equivalent_stdev(category_equivalent_stdev=50, scaler=-0.5)


def test_score_functions():
    # outliers.score_categorical_proportion()
    assert 0 <= outliers.score_categorical_proportion(value='true', 
                                                      prior_length=100, 
                                                      cdf={'true': 100}, 
                                                      scaler=0.5)['value'] <= 1
    assert 0 <= outliers.score_categorical_proportion(value='false', 
                                                      prior_length=100, 
                                                      cdf={'true': 100}, 
                                                      scaler=0.5)['value'] <= 1
    with pytest.raises(ValueError):
        outliers.score_categorical_proportion(value='false',
                                              prior_length=100,
                                              cdf={'true': 100},
                                              scaler=-0.5)
    
    # outliers.score_novelty()
    assert 0 <= outliers.score_novelty(value='19.43',
                                       cdf={'19.43': 100},
                                       scaler=0.5)['value'] <= 1
    assert 0 <= outliers.score_novelty(value='19.43',
                                       cdf={'19.86': 100},
                                       scaler=0.5)['value'] <= 1
    with pytest.raises(ValueError):
        outliers.score_novelty(value='19.43',
                               cdf={'19.86': 100},
                               scaler=-0.5)

    # outliers.score_estimated_ppf()
    assert 0 <= outliers.score_numeric_ppf(value=50,
                                           cdf={"0.0": 558.0,
                                                "1.52587890625e-05": 558.0,
                                                "0.00390625": 558.0,
                                                "0.015625": 558.0,
                                                "0.0625": 558.0,
                                                "0.25": 558.0,
                                                "0.5": 559.0,
                                                "0.75": 559.0,
                                                "0.9375": 559.0,
                                                "0.984375": 559.0,
                                                "0.99609375": 559.0,
                                                "0.9999847412109375": 559.0,
                                                "1.0": 559.0},
                                           scaler=0.5)['value'] <= 1
    with pytest.raises(ValueError):
        outliers.score_numeric_ppf(value=50,
                                   cdf={"0.0": 558.0,
                                        "1.52587890625e-05": 558.0,
                                        "0.00390625": 558.0,
                                        "0.015625": 558.0,
                                        "0.0625": 558.0,
                                        "0.25": 558.0,
                                        "0.5": 559.0,
                                        "0.75": 559.0,
                                        "0.9375": 559.0,
                                        "0.984375": 559.0,
                                        "0.99609375": 559.0,
                                        "0.9999847412109375": 559.0,
                                        "1.0": 559.0},
                                   scaler=-0.5)

    # outliers.score_dist_from_mean()
    assert 0 <= outliers.score_dist_from_mean(value=0,
                                              stdev=0,
                                              mean=1,
                                              scaler=0.5)['value'] <= 1
    with pytest.raises(ValueError):
        outliers.score_dist_from_mean(value=0,
                                      stdev=0,
                                      mean=1,
                                      scaler=-0.5)


def test_ppf_functions():
    numeric_cdf = {"0.0": 0.0,
                   "1.52587890625e-05": 1.0,
                   "0.00390625": 1.0,
                   "0.015625": 1.0,
                   "0.0625": 1.0,
                   "0.25": 70.0,
                   "0.5": 210.0,
                   "0.75": 280.0,
                   "0.9375": 401.0,
                   "0.984375": 513.0,
                   "0.99609375": 605.0,
                   "0.9999847412109375": 1715.0,
                   "1.0": 1716.0}

    assert outliers.get_numeric_ppf(210, numeric_cdf) == 0.5
    assert outliers.get_numeric_ppf(1, numeric_cdf) == 1.52587890625e-05
    assert outliers.get_numeric_ppf(270, numeric_cdf) < 0.75
    assert outliers.get_numeric_ppf(0.5, numeric_cdf) < 1.52587890625e-05
    assert outliers.get_numeric_ppf(1715.5, numeric_cdf) < 1
    assert outliers.get_numeric_ppf(1717, numeric_cdf) == 1
    assert outliers.get_numeric_ppf(-1, numeric_cdf) == 0

