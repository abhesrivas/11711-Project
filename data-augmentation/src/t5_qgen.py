from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel

import logging
import json

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)

corpus_path = "/home/ubuntu/T5QGen/gen_corpus.jsonl"
corpus = GenericDataLoader(corpus_file=corpus_path).load_corpus()

###########################
#### Query-Generation  ####
###########################

#### Model Loading
model_path = "iarfmoose/t5-base-question-generator"
generator = QGen(model=QGenModel(model_path))

#### Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
#### https://huggingface.co/blog/how-to-generate
#### Prefix is required to seperate out synthetic queries and qrels from original
prefix = "gen-3"

#### Generating 3 questions per document for all documents in the corpus 
#### Reminder the higher value might produce diverse questions but also duplicates
ques_per_passage = 3

#### Generate queries per passage from docs in corpus and save them in original corpus
#### check your datasets folder to find the generated questions, you will find below:
#### 1. datasets/scifact/gen-3-queries.jsonl
#### 2. datasets/scifact/gen-3-qrels/train.tsv

batch_size = 32

generator.generate(corpus, output_dir="/home/ubuntu/T5QGen/results", ques_per_passage=ques_per_passage, prefix=prefix, batch_size=batch_size)

generated_qs_path = "/home/ubuntu/T5QGen/results/gen-3-queries.jsonl"
generated_qs = []
with open(generated_qs_path, "r") as stream:
    generated_qs = stream.readlines()
generated_qs = [json.loads(item) for item in generated_qs]

generated_corpus_path = "/home/ubuntu/T5QGen/gen_corpus.jsonl"
with open(generated_corpus_path, "r") as stream:
    for doc_itr, doc in enumerate(stream.readlines()):
        doc_dict = json.loads(doc)
        for itr in range(doc_itr*ques_per_passage, doc_itr*ques_per_passage+ques_per_passage-1):
            try:
                generated_qs[itr].update({"metadata": doc_dict["metadata"]})
            except Exception as ex:
                print(f"error in writing corpus: {str(ex.args)}")

formatted_qs_path = "/home/ubuntu/T5QGen/results/formatted-gen-3-queries.jsonl"
with open(formatted_qs_path, "w") as stream:
    for q in generated_qs:
        stream.write(json.dumps(q)+"\n")
