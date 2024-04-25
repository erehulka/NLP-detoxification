import datetime
import re
import signal
import sys
import pandas as pd
from datasets import load_dataset
from transformers import pipeline, BertTokenizer

SPACES = re.compile(r"\s+")

toxicWords = load_dataset("textdetox/multilingual_toxic_lexicon")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input = pd.read_csv(sys.argv[1], sep='\t')
input['neutral_sentence'] = 'x'
classifier = pipeline("fill-mask")

def signal_handler(sig, frame):
  now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
  input.to_csv(f'outputs/{now}.tsv', sep='\t', index=False)
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def maskSwearWords(input: str, stopwords: list[str]) -> str:
  tokens = [
    token
    if not stopwords or token.lower().strip() not in stopwords
    else "<mask>"
    for token in tokenizer.tokenize(input)
  ]
  return tokenizer.convert_tokens_to_string(tokens)

for index, row in input.iterrows():
  if row['lang'] != 'en':
    continue
  masked = maskSwearWords(row['toxic_sentence'], toxicWords[row['lang']]["text"])
  if '<mask>' not in masked:
    input.at[index, 'neutral_sentence'] = masked
    print(index, row['toxic_sentence'], masked)
    continue
  maskPredict = classifier(masked)
  print(maskPredict)
  detoxified = maskPredict[0]['sequence']

  input.at[index, 'neutral_sentence'] = detoxified
  print(index, row['toxic_sentence'], masked, detoxified)

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/mask_{now}.tsv', sep='\t', index=False)