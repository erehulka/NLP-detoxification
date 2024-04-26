import datetime
import pandas as pd
import sys

input = pd.read_csv(sys.argv[1], sep='\t')
for index, row in input.iterrows():
  if pd.isna(row['neutral_sentence']):
    input.at[index, 'neutral_sentence'] = 'x'

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/{now}.tsv', sep='\t', index=False)