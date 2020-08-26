import pandas as pd
from shutil import copyfile
import os

df = pd.read_csv(r'C:\Users\ndowmon\data\csv-outlier-predictions-2019_08_26-13_02_04__01.csv')

bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
# labels = ['0-10', '10-20']
# df['binned'] = pd.cut(df['prediction'], bins=bins)

s = df.groupby(pd.cut(df[' prediction'].astype(float), bins=bins)).size()
print (s)
print(len(df))

src_dir = r'C:\Users\ndowmon\data\outlier-predictions-2019_08_10-10_17_12\predictions'
dst_dir = r'C:\Users\ndowmon\data\outlier-predictions-gt-10pct'

for filename in df[df[' prediction'].astype(float) > 0.1]['filename']:
    copyfile(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))
