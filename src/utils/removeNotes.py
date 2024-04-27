import datetime
import re
import sys
import pandas as pd

if len(sys.argv) != 2:
  raise Exception("Input csv file must be specified")

input = pd.read_csv(sys.argv[1], sep='\t')

for index, row in input.iterrows():
  processed = row['neutral_sentence']
  processed = re.sub(r'\(?Note:.*?$', '', processed)
  processed = re.sub(r'Output:', '', processed)
  input.at[index, 'neutral_sentence'] = processed

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/{now}_processed.tsv', sep='\t', index=False)