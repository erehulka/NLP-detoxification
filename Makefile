install:
	- mkdir outputs

create_indices:
	- python3 src/createIndex.py 'rag_index' indices/rag_index

run_rag: create_indices
	export PYTHONPATH=$(pwd)
	python3 src/rag.py test_without_answers.tsv