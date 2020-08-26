import io
import os
import json
from datetime import datetime
import sys
import shutil
import pandas as pd
import tempfile


prediction_directory = r'C:\Users\ndowmon\data\outlier-predictions-2019_08_26-13_02_04\predictions'
filename = r'C:\Users\ndowmon\data\outlier-predictions-2019_08_26-13_02_04\summary.csv'

summary_filename = filename
print(f'{datetime.now()}: Summarizing prediction scores at {summary_filename}...')
i = 0
n = len(os.listdir(prediction_directory))
print(n)
loopstart = datetime.now()
buff = io.BytesIO()
buff.write(b'filename,prediction\n')

for file in os.listdir(prediction_directory):
    i += 1
    j = (i + 1) / n
    s = (datetime.now() - loopstart).seconds * (1 / j - 1)
    time_remaining = f'{s // 3600} hours, {(s % 3600) // 60} minutes'
    sys.stdout.write(f"\r[%-20s] {i} files processed ({round(j * 100, 1)}%%, about {time_remaining} remain)" % ('=' * int(20 * j)))
    # sys.stdout.write(f"\r{i} ({round(j * 100, 1)}%%, about {time_remaining} remain)")
    sys.stdout.flush()
    with open(os.path.join(prediction_directory, file), 'r') as fp:
        jsn = json.load(fp)
        f_str = f'{file},{jsn["prediction"]}\n'
        buff.write(f_str.encode('utf-8'))

with open(summary_filename, 'wb') as fd:
    buff.seek(0)
    shutil.copyfileobj(buff, fd)

df = pd.read_csv(summary_filename)
bins = [-(10**-10), .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
s = df.groupby(pd.cut(df['prediction'].astype(float), bins=bins)).size()
print(s)
maxrow = df.loc[df['prediction'].idxmax()]
print(f'\nMAXIMUM: {maxrow["filename"]}, \tscore = {maxrow["prediction"]}')
print(i)