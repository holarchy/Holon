import common
import outliers

import pandas as pd
import os
import io
from datetime import datetime
import json
import yaml
import sys
import shutil


def predict_outliers(input_filepath, out_dir, desired_subnet, priors_parent_dir):

    # use the same file config that was used to create priors.
    file_config = common.load_file_config(priors_parent_dir)
    print('WARNING: The outliers process currently drops files into a single directory. Indexing in NTFS and other '
          'file systems cause the write process to slow down at O(n) when there are many files in the directory. '
          'Future solutions include migrating to a distributed database; in the meantime, we suggest keeping outlier '
          'processes to fewer than 100,000 rows of NetFlow data.')

    print(f'{datetime.now()}: Opening File {input_filepath}...')
    qradar_df = pd.read_csv(input_filepath, dtype=file_config.biflow_fields)

    print(f'{datetime.now()}: Making /metadata.json File...')
    with open(os.path.join(priors_parent_dir, 'metadata.json')) as fp:
        metadata = json.load(fp)
        if common.compare_versions(metadata['package version'], common.__version__ , 'minor') is False:
            raise ValueError(f'Cannot create new priors of version {common.__version__} from priors which were created '
                             f'with version {metadata["package version"]}')

    outlier_directory = os.path.join(out_dir, 'outlier-predictions-' + datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
    os.mkdir(outlier_directory)
    with open(os.path.join(outlier_directory, 'metadata.json'), 'w') as fp:
        metadata_obj = common.get_file_metadata(input_filepath, desired_subnet, priors_parent_dir)
        metadata_obj.update({'priors_dir': priors_parent_dir,
                             'package_version': common.__version__})
        json.dump(obj=metadata_obj, fp=fp)

    with open(os.path.join(outlier_directory, 'file_config.yml'), 'w') as fp:
        yaml.dump(file_config.to_yml(), fp)

    print(f'{datetime.now()}: Opening {desired_subnet}.json File...')
    priors_dir = os.path.join(priors_parent_dir, 'priors')
    subnet_prior_dir = os.path.join(priors_dir, desired_subnet)
    with open(os.path.join(subnet_prior_dir, '.json'), 'r') as fp:
        subnet_prior = json.load(fp)
    subnet_priors = {desired_subnet: subnet_prior}  # predict interface should take multiple subnet priors for consistency.

    print(f'{datetime.now()}: Making Outlier Predictions...')
    prediction_directory = os.path.join(outlier_directory, 'predictions')
    os.mkdir(prediction_directory)

    exception_directory = os.path.join(outlier_directory, 'exceptions')
    os.mkdir(exception_directory)

    i = 0
    n = len(qradar_df)
    loopstart = datetime.now()
    print(f'Creating predictions for {n} unique network flows')
    exceptions = 0
    for idx, row in qradar_df.iterrows():
        i += 1
        # sys.stdout.write('\r')
        j = (i + 1) / n
        s = (datetime.now() - loopstart).seconds * (1 / j - 1)
        time_remaining = f'{s // 3600} hours, {(s % 3600) // 60} minutes'
        sys.stdout.flush()
        sys.stdout.write(f"\r[%-20s] {i} files processed, {exceptions} exceptions ({round(j * 100, 1)}%%, about {time_remaining} remain)" % ('=' * int(20 * j)))
        my_str = file_config.uniflow_this_prfx
        ip_str = file_config.hierarchy[1]

        flow_df = common.make_prediction_df(row.to_frame().T, desired_subnet)
        if len(flow_df):
            raw_data = row.to_dict()
            ips = flow_df[my_str + ip_str].unique()
            # todo make sure this works when both machines are from the same subnet
            ip_priors = dict()
            for ip in ips:
                ip_filepath = os.path.join(subnet_prior_dir, ip, '.json')
                if os.path.exists(ip_filepath):
                    with open(ip_filepath, 'r') as fp:
                        ip_priors[ip] = json.load(fp)
            try:
                predictions = outliers.predict(flow_df, subnet_priors, ip_priors, file_config)
                dir = prediction_directory
            except Exception as e:
                exceptions += 1
                predictions = {'exception': str(e)}
                dir = exception_directory
            predictions['raw_data'] = raw_data
            with open(os.path.join(dir, str(int(row['lastpackettime'])) + '_' +
                                        row[file_config.biflow_src_prfx + ip_str] + '_' +
                                        row[file_config.biflow_dst_prfx + ip_str] + '.json'), 'w') as fp:
                json.dump(predictions, fp)

    make_summary(outlier_directory)


def make_summary(outlier_directory):
    file_config = common.load_file_config(outlier_directory)
    sys.stdout.write('\n')
    prediction_directory = os.path.join(outlier_directory, 'predictions')
    summary_filename = os.path.join(outlier_directory, 'summary.csv')
    print(f'{datetime.now()}: Summarizing prediction scores at {summary_filename}...')
    i = 0
    n = len(os.listdir(prediction_directory))
    loopstart = datetime.now()
    buff = io.BytesIO()
    buff.write(b'filename,prediction,src.ip,src.port,dst.ip,dst.port,classification\n')
    for file in os.listdir(prediction_directory):
        # print(file)
        i += 1
        j = (i + 1) / n
        s = (datetime.now() - loopstart).seconds * (1 / j - 1)
        time_remaining = f'{s // 3600} hours, {(s % 3600) // 60} minutes'
        sys.stdout.flush()
        sys.stdout.write(f"\r[%-20s] {i} files processed ({round(j * 100, 1)}%%, about {time_remaining} remain)" % ('=' * int(20 * j)))
        # sys.stdout.write(f"\r{i} ({round(j * 100, 1)}%%, about {time_remaining} remain)")
        # sys.stdout.flush()
        with open(os.path.join(prediction_directory, file), 'r') as fp:
            try:
                jsn = json.load(fp)
                prediction = jsn['prediction']
                raw_data = jsn['raw_data']
                src_ip = raw_data[file_config.biflow_src_prfx + 'ip']
                dst_ip = raw_data[file_config.biflow_dst_prfx + 'ip']
                src_port = raw_data[file_config.biflow_src_prfx + 'port']
                dst_port = raw_data[file_config.biflow_dst_prfx + 'port']
                buff.write(f'{file},{prediction},{src_ip},{src_port},{dst_ip},{dst_port},\n'.encode('utf-8'))
            except json.decoder.JSONDecodeError:
                # this usually happens if the initial process is interrupted.
                pass

    sys.stdout.write('\n')
    with open(summary_filename, 'wb') as fd:
        buff.seek(0)
        shutil.copyfileobj(buff, fd)

    df = pd.read_csv(summary_filename)
    bins = [-(10**-10), .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    s = df.groupby(pd.cut(df['prediction'].astype(float), bins=bins)).size()
    print(s)
    maxrow = df.loc[df['prediction'].idxmax()]
    print(f'\nMAXIMUM: {maxrow["filename"]}, \tscore = {maxrow["prediction"]}')


if __name__ == "__main__":
    # predict_outliers(input_filepath='/home/nick/Data/data-2019-05-08_14.csv',
    #                  out_dir='/home/nick/Data/_ensembles',
    #                  desired_subnet='19.43',
    #                  priors_parent_dir='/home/nick/Data/_ensembles/priors-2019_11_12-19_33_13')

    make_summary(outlier_directory=r'/home/nick/Data/_ensembles/outlier-predictions-2020_01_03-10_25_41')
