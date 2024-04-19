from datasets import load_dataset, Dataset

class DetoxifyEn:

  _toxicWordsDataset = load_dataset("textdetox/multilingual_toxic_lexicon")

  def __init__(self, inputSentences: Dataset) -> None:
    pass
    # This should run such thing, which will return as output for each input the detoxified output.

    # Idea - mask each toxic word (tokens), and then try to substitute it somehow (wow)