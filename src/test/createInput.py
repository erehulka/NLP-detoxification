import json
import sys
import pandas as pd

if len(sys.argv) != 2:
  raise Exception("Input csv file must be specified")

input = pd.read_csv(sys.argv[1], sep='\t')

jsonData = []
for index, row in input.iterrows():
  if index > 3202 and index < 3272:
    jsonData.append({'id': index, 'text': row['toxic_sentence']})

with open('input.jsonl', 'w', encoding='utf-8') as f:
  json.dump(jsonData, f, ensure_ascii=False, indent=4)