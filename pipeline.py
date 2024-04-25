import datetime
import pandas as pd
import sys
import signal

import requests

PROMPTS = {
  'v1.0': 'Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. The result must be in the same language as the original text. Please provide only the rewritten text with no additional info. "{phrase}"',
  'v1.1': 'Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. The result must be in the same language as the original text. Provide only the rewritten text, do not add any additional info or notes. "{phrase}"',
  'v2.0': 'You are a text de-toxifier. You receive a toxic text and your task is to re-write it in a non-toxic way while saving the main content. You do not respond anything else, just the rewritten, non-toxic text. The input is in "" and can be in any language. The result must be in the same language as the input. "{phrase}"'
}


def callLlamaApi(prompt: str) -> str:
  url = "http://localhost:11434/api/generate"
  data = {
    "model": "llama3",
    "prompt": prompt,
    "stream": False
  }

  try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
      return response.json()['response']
    else:
      print("Error: Unexpected status code:", response.status_code)
  except requests.exceptions.RequestException as e:
    print("Error: Request failed:", e)

print(callLlamaApi(PROMPTS['v1.0.0'].format(phrase="holy shit , just finished the season and it is so good but so * dark * , especially at the end .")))

if len(sys.argv) != 2:
  raise Exception("Input csv file must be specified")

input = pd.read_csv(sys.argv[1], sep='\t')
input['neutral_sentence'] = 'x'

def signal_handler(sig, frame):
  now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
  input.to_csv(f'outputs/{now}.tsv', sep='\t', index=False)
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

for index, row in input.iterrows():
  detoxified = callLlamaApi(PROMPTS['v1.0.0'].format(phrase=row['toxic_sentence']))
  if detoxified[0] == "\"":
    detoxified = detoxified[1:-1]
  input.at[index, 'neutral_sentence'] = detoxified
  print(index, row['toxic_sentence'], detoxified)

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/{now}.tsv', sep='\t', index=False)