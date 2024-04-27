import datetime
import re
import pandas as pd
import sys
import signal

from src.utils.api import callLlamaApi

LANGUAGES = {
  'am': 'Amharic',
  'ar': 'Arabic',
  'de': 'German',
  'en': 'English',
  'es': 'Spanish',
  'hi': 'Hindi',
  'ru': 'Russian',
  'uk': 'Ukrainian',
  'zh': 'Chinese'
}

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
  'v3.1': """
    You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
    Your response only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Try not to rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
    The language of the input is {lang} and the language of the response must be the same.
    Here are some examples what should be your response for given texts:
    - Input: "she is looking like shite .", your response: "she is looking like bad",
    - Input: "btw jonhson county is a real shithole in case you were wondering .", your response: "Johnson county is not good.",
    - Input: "why are canadians so taken up with hockey when their teams are all losers ?", your response: "Why are canadians so taken up woth hockey?",
    - Input: "keep crying me a river , bitch boy .", your response: "keep crying me a river , ya boy .",
    - Input: "until feminists take on islam and how it treats women , the entire movement is a total farce !", your response: "Feminism is not a good movement",
    - Input: "countless crap has been spotted by me .", your response: "countless wrong things has been spotted by me .",
    or in other languages:
    - Input (in Russian): "ну и выродки,это твой ребёнок,зачем ты сучка рожала.", your response: "ну и люди,это твой ребёнок,зачем ты рожала."

    Text to be detoxified: "{phrase}"
  """,
  'v3.1.1': """
    You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
    Output only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Try not to rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
    The language of the input is {language} and the language of the output must be the same.
    Here are some examples what should be the output for given texts:
    Input: "she is looking like shite .", 
    Output: "she is looking like bad",
    Input: "btw jonhson county is a real shithole in case you were wondering .",
    Output: "Johnson county is not good.",
    Input: "why are canadians so taken up with hockey when their teams are all losers ?",
    Output: "Why are canadians so taken up woth hockey?",
    Input: "keep crying me a river , bitch boy .", 
    Output: "keep crying me a river , ya boy .",
    Input: "until feminists take on islam and how it treats women , the entire movement is a total farce !",
    Output: "Feminism is not a good movement",
    Input: "countless crap has been spotted by me .",
    Output: "countless wrong things has been spotted by me .",
    or in other languages:
    Input (in Russian): "ну и выродки,это твой ребёнок,зачем ты сучка рожала.",
    Output "ну и люди,это твой ребёнок,зачем ты рожала."

    Provide only the detoxified text as the output, no additional information. 
    Input: "{phrase}"
  """,
  'v3.2': """
    You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
    Output only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Try not to rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
    The language of the input is {language} and the language of the output must be the same.
    Here are some examples in English what should be  the output for given texts:
    - Input: "she is looking like shite .", 
      Output: "she is looking like bad",
    - Input: "btw jonhson county is a real shithole in case you were wondering .",
      Output: "Johnson county is not good.",
    - Input: "why are canadians so taken up with hockey when their teams are all losers ?",
      Output: "Why are canadians so taken up woth hockey?",
    - Input: "keep crying me a river , bitch boy .", 
      Output: "keep crying me a river , ya boy .",
    - Input: "until feminists take on islam and how it treats women , the entire movement is a total farce !",
      Output: "Feminism is not a good movement",
    - Input: "countless crap has been spotted by me .",
      Output: "countless wrong things has been spotted by me .",
    or in other languages:
    - Input (in Russian): "ну и выродки,это твой ребёнок,зачем ты сучка рожала.",
      Output: "ну и люди,это твой ребёнок,зачем ты рожала."
    - Input (in Amharic): "ገልቱዬ ስንቴ ነው ሚሞቱት ግን? መሞታቸውን የዛሬ ወርም አርድተኘን ነበር ??",
      Output: "ገልቱዬ ስንቴ ነው ሚሞቱት ግን? ዛሬ ወርም አርደኘን ነበር ??"
    - Input (in Arabic): "ضاعت فلوسك يا دبه ببلاش",Output: "ضاعت مبلغك يا شخصاً بلبنة"
    - Input (in German): "dieser Holtmann scheint ein Grüner-Bastard zu sein! #BerichtAusBerlin",
      Output: "Dieser Holtmann scheint ein Grüner zu sein! #BerichtAusBerlin"
    - Input (in Spanish): "pues hazme los deberes furcia barata",
      Output: "Pues hazme los deberes a tiempo barata."
    - Input (in Hindi): "मैच फिनिश कर देना चाहिए    गांड पे हाथ रख कर बैठ नहीं जाना चाहिए",
      Output: "मैं सीखना चाहता हूँ    गांड पे हाथ न रहकर बैठ जाना चाहिए"
    - Input (in Ukrainian): "От нахуя мене зайобувати з самого ранку, a???",
      Output: "Від чого мене засмущує з самого ранку, a???"
    - Input (in Chinese): "在非洲孤独寂寞呗，母猪赛貂蝉",
      Output: "在非洲孤独寂寞了，母猪赛蝉"

    Provide only the detoxified text as the output, no additional information. Input (in {language}): "{phrase}"
  """,
  'v3.2.1': """
    You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
    Output only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Try not to rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
    The language of the input is {language} and the language of the output must be the same.
    Here are some examples in English what should be  the output for given texts:
    Input: "she is looking like shite .", 
    Output: "she is looking like bad",
    Input: "btw jonhson county is a real shithole in case you were wondering .",
    Output: "Johnson county is not good.",
    Input: "why are canadians so taken up with hockey when their teams are all losers ?",
    Output: "Why are canadians so taken up woth hockey?",
    Input: "keep crying me a river , bitch boy .", 
    Output: "keep crying me a river , ya boy .",
    Input: "until feminists take on islam and how it treats women , the entire movement is a total farce !",
    Output: "Feminism is not a good movement",
    Input: "countless crap has been spotted by me .",
    Output: "countless wrong things has been spotted by me .",
    or in other languages:
    Input (in Russian): "ну и выродки,это твой ребёнок,зачем ты сучка рожала.",
    Output: "ну и люди,это твой ребёнок,зачем ты рожала."
    Input (in Amharic): "ገልቱዬ ስንቴ ነው ሚሞቱት ግን? መሞታቸውን የዛሬ ወርም አርድተኘን ነበር ??",
    Output: "ገልቱዬ ስንቴ ነው ሚሞቱት ግን? ዛሬ ወርም አርደኘን ነበር ??"
    Input (in Arabic): "ضاعت فلوسك يا دبه ببلاش",
    Output: "ضاعت مبلغك يا شخصاً بلبنة"
    Input (in German): "dieser Holtmann scheint ein Grüner-Bastard zu sein! #BerichtAusBerlin",
    Output: "Dieser Holtmann scheint ein Grüner zu sein! #BerichtAusBerlin"
    Input (in Spanish): "pues hazme los deberes furcia barata",
    Output: "Pues hazme los deberes a tiempo barata."
    Input (in Hindi): "मैच फिनिश कर देना चाहिए    गांड पे हाथ रख कर बैठ नहीं जाना चाहिए",
    Output: "मैं सीखना चाहता हूँ    गांड पे हाथ न रहकर बैठ जाना चाहिए"
    Input (in Ukrainian): "От нахуя мене зайобувати з самого ранку, a???",
    Output: "Від чого мене засмущує з самого ранку, a???"
    Input (in Chinese): "在非洲孤独寂寞呗，母猪赛貂蝉",
    Output: "在非洲孤独寂寞了，母猪赛蝉"

    Provide only the detoxified text as the output, no additional information or notes. 
    Input (in {language}): "{phrase}"
  """
}

print(callLlamaApi(PROMPTS['v3.2.1'].format(phrase="holy shit , just finished the season and it is so good but so * dark * , especially at the end .", language=LANGUAGES['en'])))

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
  detoxified = callLlamaApi(PROMPTS['v3.2.1'].format(phrase=row['toxic_sentence'], language=LANGUAGES[row['lang']]))
  if detoxified[0] == "\"":
    detoxified = detoxified[1:-1]

  input.at[index, 'neutral_sentence'] = detoxified
  print(index, row['toxic_sentence'], detoxified)

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/{now}.tsv', sep='\t', index=False)