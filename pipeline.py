import datetime
import pandas as pd
import sys
import signal

from utils.api import callLlamaApi

PROMPTS = {
  'v1.0': 'Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. The result must be in the same language as the original text. Please provide only the rewritten text with no additional info. "{phrase}"',
  'v1.1': 'Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. The result must be in the same language as the original text. Provide only the rewritten text, do not add any additional info or notes. "{phrase}"',
  'v1.2': 'Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. Do not add more context, if the text is short, keep it short. The result must be in the same language as the original text. Provide only the rewritten text, do not add any additional info or notes. "{phrase}"',
  'v1.3': 'Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. Rewrite just the toxic part, keep the non-toxic parts the same. Do not add more context, if the text is short, keep it short. The result must be in the same language as the original text. Provide only the rewritten text, do not add any additional info or notes. "{phrase}"',
  'v1.4': 'Detoxify this text. Respond only with the original text but detoxified, nothing else like "Here is the detoxified text:" etc.: {phrase}',
  'v2.0': 'You are a text de-toxifier. You receive a toxic text and your task is to re-write the toxic parts in a non-toxic way while saving the main content. Do not add more text, if the text is short, keep it short. Keep the non-toxic words in the text the same. You do not respond anything else, just the rewritten, non-toxic text. The input is in "" and can be in any language. The result must be in the same language as the input. "{phrase}"',
  'vmask_tesk': 'Mark toxic parts in this sentence as <mask>: {phrase}',
}

print(callLlamaApi(PROMPTS['v1.4'].format(phrase="holy shit , just finished the season and it is so good but so * dark * , especially at the end .")))

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
  if row['lang'] != 'en':
    continue
  detoxified = callLlamaApi(PROMPTS['v1.4'].format(phrase=row['toxic_sentence']))
  if detoxified[0] == "\"":
    detoxified = detoxified[1:-1]
  input.at[index, 'neutral_sentence'] = detoxified
  print(index, row['toxic_sentence'], detoxified)

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/{now}.tsv', sep='\t', index=False)