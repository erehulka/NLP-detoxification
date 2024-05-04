import datetime
import signal
import sys
from chromadb.utils import embedding_functions
import chromadb
import pandas as pd

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

PROMPT = """
You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
Output only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Do not rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
The language of the input is {language} and the language of the response must be the same.
Here are some examples what should be the output for given texts:
{examples}


Your output is only the detoxified text, you do not say anything else.

Input: {phrase}
"""

if len(sys.argv) != 2:
  raise Exception("Input csv file must be specified")

input = pd.read_csv(sys.argv[1], sep='\t')
input['neutral_sentence'] = 'x'

def signal_handler(sig, frame):
  now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
  input.to_csv(f'outputs/rag_{now}.tsv', sep='\t', index=False)
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

index_name = "rag_index"
index_path = "indices/rag_index"
embedding_model = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=str(index_path))
collection = client.get_collection(name=index_name)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

for index, row in input.iterrows():
  results = collection.query(query_embeddings=embedding_fn([row['toxic_sentence']]), n_results=6)
  examples = ""
  for text, metadata in zip(*results['documents'], *results['metadatas']):
      examples += f"Input: {text}\n\nPrediction: {metadata['neutral_sentence']}\n\n"

  finalPrompt = PROMPT.format(
    language=LANGUAGES[row['lang']],
    examples=examples,
    phrase=row['toxic_sentence']
  )
  detoxified = callLlamaApi(finalPrompt)
  if detoxified[0] == "\"" or detoxified[0] == '“' or detoxified[0] == '«':
    detoxified = detoxified[1:-1]

  input.at[index, 'neutral_sentence'] = detoxified
  print(index, row['toxic_sentence'], detoxified)

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/rag_{now}.tsv', sep='\t', index=False)