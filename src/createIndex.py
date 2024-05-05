import typer
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset

def main(index_name: str, index_path: Path, embedding_model: str = "sentence-transformers/LaBSE"):
    datasetDict: DatasetDict = load_dataset("textdetox/multilingual_paradetox")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    i = 1
    for lang, dataset in tqdm(datasetDict.items()):
        index_path_full = Path(str(index_path) + '_' + lang)
        index_name_full = index_name + '_' + lang
        client = chromadb.PersistentClient(path=str(index_path_full))
        collection = client.create_collection(name=index_name_full)
        dataset: Dataset = dataset
        for row in dataset:
            collection.add(
                embeddings=embedding_fn([row["toxic_sentence"]]),
                documents=[row["toxic_sentence"]],
                metadatas=[{"neutral_sentence": row["neutral_sentence"]}],
                ids=[str(i)]
            )
            i += 1

if __name__ == "__main__":
    typer.run(main)