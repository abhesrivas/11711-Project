from elasticsearch import Elasticsearch
import json
import pdb

es = Elasticsearch()

with open("/home/ubuntu/ANLP/relevance-IR/data/beir/corpus.jsonl", "r") as f:
    data = f.readlines()

    for i,d in enumerate(data):
        hashmap = json.loads(d)
        # hashmap = hashmap["title"] + " " + hashmap["text"]
        es.index(index="corpus-test", id=i, document=hashmap)

# es.indices.delete(index="corpus-test")