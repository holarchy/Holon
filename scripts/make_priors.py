import priors
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                    help='Raw flow data in .csv format.', type=str, required=True)
parser.add_argument('-d', '--out_directory',
                    help='Output directory; priors will end up here.', type=str, required=True)
parser.add_argument('-s', '--subnet',
                    help='Subnet to create prior from.', default='19.43')

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

desired_subnet = args.subnet

priors.make_priors(input_filepath, out_dir, desired_subnet)
