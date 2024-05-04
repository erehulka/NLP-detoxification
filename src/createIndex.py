import typer
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset

def main(index_name: str, index_path: Path, embedding_model: str = "all-MiniLM-L6-v2"):
    datasetDict: DatasetDict = load_dataset("textdetox/multilingual_paradetox")

    client = chromadb.PersistentClient(path=str(index_path))
    collection = client.create_collection(name=index_name)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    i = 1
    for _, dataset in tqdm(datasetDict.items()):
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