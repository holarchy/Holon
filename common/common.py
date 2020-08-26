import os
import common
from pytz import timezone
from datetime import datetime
import pandas as pd
import hashlib
import yaml
from copy import deepcopy

common_dir = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(common_dir, 'config.yml')

types_map = {'categorical': str,
             'float': float,
             'bool': bool,
             'datetime': datetime}


def validate_headers(df:pd.DataFrame, expected_headers:set):
    """

    :param df:
    :param expected_headers:
    :return: None
    """

    headers = set(df.columns.tolist())
    missing_headers = set(expected_headers) - headers
    additional_headers = headers - set(expected_headers)  # tell user any additional fields which could cause issues.
    if len(additional_headers):
        print(f'Found additional headers in supplied file: {additional_headers}. This may potentially cause issues.')
    if len(missing_headers) != 0:
        raise ValueError(f'Dataframe is missing expected headers: {missing_headers}')

    fields_with_missing_vals = []
    for header in expected_headers:
        missing_vals = df[header].isnull().values.sum()
        if missing_vals > 0:
            fields_with_missing_vals.append(header)
    if len(fields_with_missing_vals):
        print(f'Fields {fields_with_missing_vals} are missing values. This may potentially cause issues.')


def make_prediction_df(qradar_df:pd.DataFrame, desired_subnet:str=None):
    """

    :param qradar_df: DataFrame
    :param desired_subnet: str
    :return: my_perspective_network_df: DataFrame
    """
    # validate_headers(qradar_df, set(FILE_CONFIG.biflow_fields))
    if desired_subnet is None:
        raise ValueError('Please provide a "desired_subnet" parameter (ex. 19.43)')

    # re-work source only dataframe
    src_subnet_field = FILE_CONFIG.biflow_src_prfx + FILE_CONFIG.hierarchy[0]
    df_source = qradar_df[qradar_df[src_subnet_field] == desired_subnet]
    df_source.columns = df_source.columns.str.replace(FILE_CONFIG.biflow_src_prfx, FILE_CONFIG.uniflow_this_prfx)
    df_source.columns = df_source.columns.str.replace(FILE_CONFIG.biflow_dst_prfx, FILE_CONFIG.uniflow_that_prfx)
    df_source = df_source.assign(**{FILE_CONFIG.uniflow_indicator: True})

    dst_subnet_field = FILE_CONFIG.biflow_dst_prfx + FILE_CONFIG.hierarchy[0]
    df_dest = qradar_df[qradar_df[dst_subnet_field] == desired_subnet]
    df_dest.columns = df_dest.columns.str.replace(FILE_CONFIG.biflow_src_prfx, FILE_CONFIG.uniflow_that_prfx)
    df_dest.columns = df_dest.columns.str.replace(FILE_CONFIG.biflow_dst_prfx, FILE_CONFIG.uniflow_this_prfx)
    df_dest = df_dest.assign(**{FILE_CONFIG.uniflow_indicator: False})

    my_perspective_network_df = pd.concat([df_source, df_dest], sort=True).sort_index()

    # my_perspective_network_df['mytotheirbytesratio'] = my_perspective_network_df['mybytes'] / (my_perspective_network_df['mybytes'] + my_perspective_network_df['theirbytes'])
    # my_perspective_network_df['myserverfromford'] = my_perspective_network_df['mysubnet'].str.startswith('19.')
    # my_perspective_network_df['theirserverfromford'] = my_perspective_network_df['theirsubnet'].str.startswith('19.')

    # validate_headers(my_perspective_network_df, FILE_CONFIG.uniflow_fields.keys())

    return my_perspective_network_df


def md5(fname):
    """

    :param fname:
    :return:
    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_metadata(filepath, desired_subnet=None, priors_dir=None):
    now = datetime.now()
    qradar_df = pd.read_csv(filepath, dtype=FILE_CONFIG.biflow_fields)

    digest = md5(filepath)
    size_gb = os.path.getsize(filepath) / 1024 / 1024 / 1024
    start_date = datetime.fromtimestamp(qradar_df['lastpackettime'].min() / 1000, tz=timezone('UTC'))
    end_date = datetime.fromtimestamp(qradar_df['lastpackettime'].max() / 1000, tz=timezone('UTC'))
    metadata = {'md5': digest,
                'filename': filepath,
                'size (GB)': size_gb,
                'number of rows': len(qradar_df),
                'start date': str(start_date),
                'end date': str(end_date),
                'package version': common.__version__,
                'outlier prediction date': str(now)}

    if priors_dir is not None:
        metadata['priors_directory'] = priors_dir
    if desired_subnet is not None:
        metadata['subnet used'] = desired_subnet

    return metadata


def compare_versions(version_a:str, version_b:str, equivalence:str = 'minor'):
    """

    :param version_a: version like [major].[minor].[revision]
    :param version_b: version like [major].[minor].[revision]
    :param equivalence: either major, minor, or revision. default minor
    :return:
    """

    maj_a, min_a, rev_a = version_a.split('.')
    maj_b, min_b, rev_b = version_b.split('.')

    equivalent = maj_a == maj_b
    if equivalence == 'major':
        return equivalent

    equivalent = equivalent and min_a == min_b
    if equivalence == 'minor':
        return equivalent

    equivalent = equivalent and rev_a == rev_b
    if equivalence == 'revision':
        return equivalent

    raise ValueError(f'Comparing version equivalence requires either "major", "minor" or "revision" equivalence param. '
                      f'Got {equivalence}.')


class FileConfig():

    uniflow_this_prfx = 'my'
    uniflow_that_prfx = 'other'
    uniflow_indicator = 'is_source'

    def __init__(self, nondirectional_fields=None, directional_fields=None, biflow_dst_prfx=None, biflow_src_prfx=None, hierarchy=None):

        if nondirectional_fields is None and directional_fields is None:
            raise ValueError(f'Must provide either nondirectional or directional fields to {self.__class__.__name__}, '
                             f'got neither.')
        if directional_fields is not None:
            if biflow_src_prfx is None or biflow_dst_prfx is None:
                raise ValueError(f'If providing directional fields to {self.__class__.__name__}, must provide '
                                 f'biflow_src_prfx and biflow_dst_prfx prefix values. Got directional fields '
                                 f'{directional_fields}, and biflow_src_prfx = {biflow_src_prfx} and biflow_dst_prfx = '
                                 f'{biflow_dst_prfx}.')

        # map {'f1': 'str', 'f2': 'bool', 'f3': 'int'} -> {'f1': str, 'f2': bool, 'f3': int}
        nondirectional_fields = self._remap_fields(nondirectional_fields)
        directional_fields = self._remap_fields(directional_fields)

        if hierarchy is not None:
            if any([val not in directional_fields.keys() for val in hierarchy]):
                raise ValueError(f'If providing a field hierarchy to {self.__class__.__name__}, each field must '
                                 f'be a unique value represented in the directional fields. Got directional fields '
                                 f'{directional_fields.keys()} and hierarchy {hierarchy}.')
            if any([val not in [str, bool] for val in [directional_fields[k] for k in hierarchy]]):
                raise ValueError(f'If providing a field hierarchy to {self.__class__.__name__}, each field must '
                                 f'map as either a bool or str type. Got {directional_fields} and hierarchy {hierarchy}.')
        if hierarchy != ['subnet', 'ip']:
            raise ValueError(f'Currently Priors and Outliers only function with hierarcy "["subnet","ip"]". '
                             f'Got {hierarchy}.')

        self._nondirectional_fields = nondirectional_fields
        self._directional_fields = directional_fields
        self.biflow_src_prfx = biflow_src_prfx
        self.biflow_dst_prfx = biflow_dst_prfx
        self.hierarchy = hierarchy

    @staticmethod
    def _remap_fields(fields):
        if fields is None:
            fields = dict()
        else:
            for k, v in fields.items():
                fields[k] = types_map[v]
        return fields

    @staticmethod
    def _encode_fields(fields):
        encoded_fields = dict()
        inv_map = {v: k for k, v in types_map.items()}
        if len(fields) == 0:
            return None
        else:
            for k, v in fields.items():
                encoded_fields.update({k: inv_map[v]})
        return encoded_fields

    @property
    def biflow_fields(self):
        fields = dict()
        fields.update(self._nondirectional_fields)
        for field, typ in self._directional_fields.items():
            fields.update({self.biflow_src_prfx + field: typ,
                           self.biflow_dst_prfx + field: typ})
        return fields

    @property
    def uniflow_fields(self):
        fields = dict()
        fields.update(self._nondirectional_fields)
        for field, typ in self._directional_fields.items():
            fields.update({self.uniflow_this_prfx + field: typ,
                           self.uniflow_that_prfx + field: typ})
        fields.update({self.uniflow_indicator: bool})  # add "is_src" field.
        return fields

    def convert_to_uniflow(self, dataframe: pd.DataFrame):

        biflow_src_data = deepcopy(dataframe)
        biflow_src_data.columns = biflow_src_data.columns.str.replace(self.biflow_src_prfx, self.uniflow_this_prfx)
        biflow_src_data.columns = biflow_src_data.columns.str.replace(self.biflow_dst_prfx, self.uniflow_that_prfx)

        biflow_dst_data = deepcopy(dataframe)
        biflow_dst_data.columns = biflow_dst_data.columns.str.replace(self.biflow_dst_prfx, self.uniflow_this_prfx)
        biflow_dst_data.columns = biflow_dst_data.columns.str.replace(self.biflow_src_prfx, self.uniflow_that_prfx)

        biflow_src_data[self.uniflow_indicator] = True
        biflow_dst_data[self.uniflow_indicator] = False

        uniflow_data = pd.concat([biflow_src_data, biflow_dst_data])
        return uniflow_data

    def my_direction(self, field: str):
        if field.startswith(self.biflow_src_prfx):
            return self.biflow_src_prfx
        elif field.startswith(self.biflow_dst_prfx):
            return self.biflow_dst_prfx
        else:
            raise ValueError

    def their_direction(self, field: str):
        if self.my_direction(field) == self.biflow_src_prfx:
            return self.biflow_dst_prfx
        elif self.my_direction(field) == self.biflow_dst_prfx:
            return self.biflow_src_prfx
        else:
            raise ValueError(f'Could not set their_direction')

    def validate_headers(self, headers: list):
        expected_headers = self.biflow_fields.keys()
        headers = set(headers)
        missing_headers = set(expected_headers) - headers
        additional_headers = headers - set(
            expected_headers)  # tell user any additional fields which could cause issues.
        if len(additional_headers):
            print(
                f'Found additional headers in supplied file: {additional_headers}. This may potentially cause issues.')
        if len(missing_headers) != 0:
            raise ValueError(f'Dataframe is missing expected headers: {missing_headers}')

    def to_yml(self):
        return {'nondirectional_fields': self._encode_fields(self._nondirectional_fields),
                'directional_fields': self._encode_fields(self._directional_fields),
                'biflow_src_prfx': self.biflow_src_prfx,
                'biflow_dst_prfx': self.biflow_dst_prfx,
                'hierarchy': self.hierarchy}


def load_file_config(dir):
    file_config_path = os.path.join(dir, 'file_config.yml')
    with open(file_config_path, 'r') as f:
        file_config_obj = yaml.safe_load(f)

    return FileConfig(**file_config_obj)


FILE_CONFIG = load_file_config(common_dir)


def update_file_config(nondirectional_fields, directional_fields, biflow_src_prfx, biflow_dst_prfx, hierarchy):
    global FILE_CONFIG
    FILE_CONFIG = FileConfig(nondirectional_fields=nondirectional_fields,
                             directional_fields=directional_fields,
                             biflow_src_prfx=biflow_src_prfx,
                             biflow_dst_prfx=biflow_dst_prfx,
                             hierarchy=hierarchy)

    file_config_path = os.path.join(common_dir, 'file_config.yml')

    with open(file_config_path, 'w') as f:
        yaml.dump(FILE_CONFIG.to_yml(), f)


def numeric_vars():
    return [field for field, typ in FILE_CONFIG.uniflow_fields.items() if typ == float]


def categorical_vars():
    return [field for field, typ in FILE_CONFIG.uniflow_fields.items() if typ == str]


def binary_vars():
    return [field for field, typ in FILE_CONFIG.uniflow_fields.items() if typ == bool]


class Config():
    def __init__(self, database, explorer_app):
        self.database_path = database['path']
        self.outliers_path = explorer_app['outliers_path']


with open(config_path) as f:
    config_obj = yaml.safe_load(f)
CONFIG = Config(**config_obj)
