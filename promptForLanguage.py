import datetime
import signal
import sys

import pandas as pd

from src.utils.api import callLlamaApi


PROMPTS = {
  'en': """
    You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
    Output only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Try not to rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
    Here are some examples what should be the output for given texts:
    Input: "then all of a sudden i see her , shes now got the big phony tits and everything .", 
    Output: "All of a sudden i see her, she is all grown up.",
    Input: "My page should be protected first so that worthless scum like you can't keep vandalizing it.",
    Output: "My page should be protected first so that unpleasant people like you can't keep vandalizing it.",
    Input: "You made a mistake you ass.",
    Output: "You made a mistake.",
    Input: "you know more than these idiots , stay the course and the truth will prevail .", 
    Output: "you know more than these people , stay the course and the truth will prevail .",
    Input: "piss me off , fuckin jerk , get on my nerves .",
    Output: "get on my nerves",
    Input: "In addition, I see no sock puppet, jackass.",
    Output: "In addition, I see no sock puppet.",

    Provide only the detoxified text as the output, no additional information or notes. 
    Input: "{phrase}"
  """,
  'zh': """
    你是一名文本解毒员。在输入时，您会收到一段可能有毒或有害的文字。你的任务是在保留原文内容和上下文的前提下，以无害或无毒的方式改写这段文字。
    输出只包含解毒后的文本，不包含注释或其他信息。您不能在生成的文本中添加任何原文中没有的上下文。尽量不要对原文进行过多改写，只需将有毒部分完全删除，或在保留含义和上下文的前提下用一些无毒词语进行替换。
    下面是一些给定文本输出结果的示例：
    输入： “尽管浪，谁接盘谁是狗，不得house”、 
    输出： “尽管浪，谁接盘谁是狗，不得好下场”、
    输入： “对邻国的复仇情节，让日本饮鸩止渴，甚至让默代表日本人，打破了日本保持民族纯洁性的最后一点自尊打破了印象中日本对于默严格管控的神话泡沫归根结底是怕邻国报复的自我矮化心理”、
    输出： “对邻国的复仇情节，让日本饮鸩止渴，甚至让黑人代表日本人，打破了日本保持民族纯洁性的最后一点自尊打破了印象中日本对于黑人严格管控的神话泡沫归根结底是怕邻国报复的自我矮化心理”、
    输入： “重点是视频里还不是同一个默囧”、
    输出： “重点是视频里还不是同一个黑人囧”、
    输入： “这种男的就活该被戴绿帽”、 
    输出： “这种男的被背叛是不值得同情的”、
    输入： “建议仙女结婚前，男性一定要把怀胎十月分娩和坐月子的痛苦都体验一遍，这十个月一定要由她们照顾男性，以此作为她们生育时被男性对待的标准。”、
    输出： “建议自我感觉过于良好的女性结婚前，男性一定要把怀胎十月分娩和坐月子的痛苦都体验一遍，这十个月一定要由她们照顾男性，以此作为她们生育时被男性对待的标准。”、
    输入： “只准异恋普信，不准同恋自大——二极管”、
    输出： “只准异性恋没有自知之明，不准同性恋自大——二极管”、

    只提供解毒后的文本作为输出，不提供其他信息或注释。
    输入：“{phrase}”
  """,
  # 'am': 'Amharic',
  # 'ar': 'Arabic',
  # 'de': 'German',
  # 'es': 'Spanish',
  # 'hi': 'Hindi',
  # 'ru': 'Russian',
  # 'uk': 'Ukrainian',
}

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
  if row['lang'] not in PROMPTS:
    continue
  detoxified = callLlamaApi(PROMPTS[row['lang']].format(phrase=row['toxic_sentence']))
  if detoxified[0] == "\"":
    detoxified = detoxified[1:-1]

  input.at[index, 'neutral_sentence'] = detoxified
  print(index, row['toxic_sentence'], detoxified)

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/pr_for_lang_{now}.tsv', sep='\t', index=False)