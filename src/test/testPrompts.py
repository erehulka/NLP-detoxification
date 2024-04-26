from cmd import PROMPT
import datetime
import json
import pandas as pd
import sys

from src.utils.api import callLlamaApi

PROMPTS = {
  'v1.0': 'Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. The result must be in the same language as the original text. Please provide only the rewritten text with no additional info. "{phrase}"',
  'v1.1': 'Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. The result must be in the same language as the original text. Provide only the rewritten text, do not add any additional info or notes. "{phrase}"',
  'v1.2': 'Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. Do not add more context, if the text is short, keep it short. The result must be in the same language as the original text. Provide only the rewritten text, do not add any additional info or notes. "{phrase}"',
  'v1.3': 'Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. Rewrite just the toxic part, keep the non-toxic parts the same. Do not add more context, if the text is short, keep it short. The result must be in the same language as the original text. Provide only the rewritten text, do not add any additional info or notes. "{phrase}"',
  'v1.4': 'Detoxify this text. Respond only with the original text but detoxified, nothing else like "Here is the detoxified text:" etc.: {phrase}',
  'v2.0': 'You are a text de-toxifier. You receive a toxic text and your task is to re-write the toxic parts in a non-toxic way while saving the main content. Do not add more text, if the text is short, keep it short. Keep the non-toxic words in the text the same. You do not respond anything else, just the rewritten, non-toxic text. The input is in "" and can be in any language. The result must be in the same language as the input. "{phrase}"',
  'v3.0': """
    You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
    Your response only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Try not to rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
    Here are some examples what should be your response for given texts:
    - Input: "she is looking like shite .", your response: "she is looking like bad",
    - Input: "btw jonhson county is a real shithole in case you were wondering .", your response: "Johnson county is not good.",
    - Input: "why are canadians so taken up with hockey when their teams are all losers ?", your response: "Why are canadians so taken up woth hockey?",
    - Input: "keep crying me a river , bitch boy .", your response: "keep crying me a river , ya boy .",
    - Input: "until feminists take on islam and how it treats women , the entire movement is a total farce !", your response: "Feminism is not a good movement",
    - Input: "countless crap has been spotted by me .", your response: "countless wrong things has been spotted by me ."

    Text to be detoxified: "{phrase}"
  """,
}

if len(sys.argv) != 2:
  raise Exception("Input csv file must be specified")

input = pd.read_json(sys.argv[1])

for key, prompt in PROMPTS.items():
  output = []
  for index, row in input.iterrows():
    detoxified = callLlamaApi(prompt.format(phrase=row['text']))
    if detoxified[0] == "\"":
      detoxified = detoxified[1:-1]
    output.append({'id': row['id'], 'text': detoxified})

  now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
  with open('{key}.jsonl', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)