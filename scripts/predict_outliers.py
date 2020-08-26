import os
import argparse
import outliers

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                    help='Raw flow data in .csv format.', type=str, required=True)
parser.add_argument('-d', '--out_directory',
                    help='Output directory; priors will end up here.', type=str, required=True)
parser.add_argument('-s', '--subnet',
                    help='Subnet to create prior.', default='19.43')
parser.add_argument('-p', '--priors_directory',
                    help='Priors dierctory; created by make_priors.py.', required=True)

args = parser.parse_args()

## ----------------------------------- parameters --------------------------------------------
if os.path.isfile(args.file) and args.file.endswith('.csv'):
    input_filepath = args.file
else:
    raise ValueError(f'Provided file {args.file} was not a valid .csv file.')

if os.path.isdir(args.out_directory):
    out_dir = args.out_directory
else:
    raise ValueError(f'Provided output directory {args.out_directory} does not exist.')

if os.path.isdir(args.priors_directory):
    priors_dir = args.priors_directory
else:
    raise ValueError(f'Provided priors directory {args.priors_directory} does not exist.')


desired_subnet = args.subnet

outliers.predict_outliers(input_filepath, out_dir, desired_subnet, priors_dir)