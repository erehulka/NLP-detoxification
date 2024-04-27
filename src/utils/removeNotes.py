import datetime
import re
import sys
import pandas as pd

if len(sys.argv) != 2:
  raise Exception("Input csv file must be specified")

input = pd.read_csv(sys.argv[1], sep='\t')

for index, row in input.iterrows():
  input.at[index, 'neutral_sentence'] = re.sub(r'\(?Note:.*?$', '', row['neutral_sentence'])

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/{now}_processed.tsv', sep='\t', index=False)