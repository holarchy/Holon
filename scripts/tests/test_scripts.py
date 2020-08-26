import subprocess
import tempfile
import os

scripts_tests_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(scripts_tests_dir)
project_dir = os.path.dirname(scripts_dir)

redacted_filename = os.path.join(scripts_tests_dir, 'test_data', 'test_data_redacted.csv')
unredacted_filename = os.path.join(scripts_tests_dir, 'test_data', 'test_data_unredacted.csv')

redacted_priors_dir = os.path.join(scripts_tests_dir, 'test_data', 'test_priors_redacted')
unredacted_priors_dir = os.path.join(scripts_tests_dir, 'test_data', 'test_priors_unredacted')


def test_path_names():
    assert scripts_tests_dir.endswith('tests')
    assert scripts_dir.endswith('scripts')


def test_make_priors_scripts():

    make_priors_path = os.path.join(scripts_dir, 'make_priors.py')

    with tempfile.TemporaryDirectory() as temp_dir:
        # short args, redacted
        return_code = subprocess.call(f'python {make_priors_path} '
                                      f'-f {redacted_filename} '
                                      f'-d {temp_dir} '
                                      f'-s 19.43', shell=True)
        assert return_code == 0

    with tempfile.TemporaryDirectory() as temp_dir:
        # long args, redacted
        return_code = subprocess.call(f'python {make_priors_path} '
                                      f'--file {redacted_filename} '
                                      f'--out_directory {temp_dir} '
                                      f'--subnet 19.43', shell=True)
        assert return_code == 0


def test_predict_outliers_scripts():

    predict_outliers_path = os.path.join(scripts_dir, 'predict_outliers.py')

    with tempfile.TemporaryDirectory() as temp_dir:
        # short args, redacted
        return_code = subprocess.call(f'python {predict_outliers_path} '
                                      f'-f {redacted_filename} '
                                      f'-d {temp_dir} '
                                      f'-s 19.43 '
                                      f'-p {redacted_priors_dir} ', shell=True)
        assert return_code == 0

    with tempfile.TemporaryDirectory() as temp_dir:
        # long args, redacted
        return_code = subprocess.call(f'python {predict_outliers_path} '
                                      f'--file {redacted_filename} '
                                      f'--out_directory {temp_dir} '
                                      f'--subnet 19.43 '
                                      f'--priors_directory {redacted_priors_dir} ', shell=True)
        assert return_code == 0
