import traceback
import priors
import common
import pandas as pd
import os
from datetime import datetime
import json
import yaml
import sys


def make_priors(input_filepath, out_dir, desired_subnet):
    now = datetime.now()
    ## ----------------------------------- code --------------------------------------------
    print(f'{datetime.now()}: Opening File {input_filepath}...')
    output_directory = os.path.join(out_dir, 'priors-' + datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
    os.mkdir(output_directory)
    qradar_df = pd.read_csv(input_filepath, dtype=common.FILE_CONFIG.biflow_fields)

    print(f'{datetime.now()}: Validating Headers...')
    common.validate_headers(qradar_df, common.FILE_CONFIG.biflow_fields)

    print(f'{datetime.now()}: Making /metadata.json File...')
    my_perspective_df = common.make_prediction_df(qradar_df, desired_subnet)
    priors_directory = os.path.join(output_directory, 'priors')
    os.mkdir(priors_directory)
    exception_directory = os.path.join(output_directory, 'exceptions')
    os.mkdir(exception_directory)
    with open(os.path.join(output_directory, 'metadata.json'), 'w') as fp:
        metadata = common.get_file_metadata(input_filepath, desired_subnet)
        json.dump(obj=metadata, fp=fp)

    with open(os.path.join(output_directory, 'file_config.yml'), 'w') as fp:
        yaml.dump(common.FILE_CONFIG.to_yml(), fp)

    make_priors_recursively(desired_subnet, my_perspective_df, priors_directory, exception_directory,
                            common.FILE_CONFIG.hierarchy)
    print(f'\n{datetime.now()}: Priors creation successful.')


def make_priors_recursively(name, df, priors_directory, exception_directory, hierarchy):
    field_name = hierarchy[0]
    rest = hierarchy[1:]
    print(f'\n{datetime.now()}: Making /{field_name}/.json File...')

    my_field = common.FILE_CONFIG.uniflow_this_prfx + field_name
    n = len(df[my_field].unique())
    loopstart = datetime.now()
    print(f'{datetime.now()}: Creating priors for {n} unique {my_field}')
    exceptions_count = 0
    i = 0
    for val in df[my_field].unique():
        sub_df = df[df[my_field] == val]
        field_dir = os.path.join(priors_directory, val)
        os.mkdir(field_dir)

        try:
            jsn = priors.make_prior_from_df(sub_df)
            d = field_dir
            path = '.json'
        except Exception as e:
            jsn = {'traceback': traceback.format_exc()}
            d = exception_directory
            path = val + '.json'

        with open(os.path.join(d, path), 'w') as fp:
            json.dump(jsn, fp)

        i += 1
        sys.stdout.write('\r')
        j = (i + 1) / n
        s = (datetime.now() - loopstart).seconds * (1 / j - 1)
        time_remaining = f'{s // 3600} hours, {(s % 3600) // 60} minutes'
        sys.stdout.write(f"[%-20s] {i} files processed, {exceptions_count} exceptions ({round(j*100,1)}%%, about {time_remaining} remain)" % ('='*int(20*j)))
        sys.stdout.flush()

        if len(rest):
            make_priors_recursively(val, sub_df, field_dir, exception_directory, rest)


if __name__ == "__main__":
    make_priors(r'/home/nick/Data/data-2019-05-01_07-old-headers.csv',
                r'/home/nick/Data/_ensembles', '19.43')
