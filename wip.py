from datasets import load_dataset

dataset = load_dataset("textdetox/multilingual_toxic_lexicon")

print(dataset['en'][0])