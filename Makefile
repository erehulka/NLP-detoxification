install:
	- mkdir outputs

create_indices:
	- rm -rf indices
	python3 src/createIndex.py 'rag_index' indices/rag_index