import datetime
import pandas as pd
import sys

if len(sys.argv) != 2:
  raise Exception("Input csv file must be specified")

input = pd.read_csv(sys.argv[1], sep='\t')
input['neutral_sentence'] = input.get('neutral_sentence', ['' for _ in range(len(input))]) 

for index, row in input.iterrows():
  if row['lang'] != 'en':
    continue

  input.at[index, 'neutral_sentence'] = 'TODO' # TODO result of the pipeline.

  
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/{now}.tsv', sep='\t', index=False)