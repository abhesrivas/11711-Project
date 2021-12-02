import json
import pathlib, os
import logging
import copy

import numpy as np

from random import sample
from tqdm import tqdm
from pprint import pprint

all_docs_path = "/home/ubuntu/Karthik/11711-Project/relevance-IR/data/beir-format/corpus.jsonl"
generated_corpus_path = "/home/ubuntu/T5QGen/gen_corpus.jsonl"

lbound, ubound = 5, 15
sents_per_doc = 25
t5_qgen_schema = "answer_token {} context_token {}"
offset = 10

gen_sent_idx = 0
sentences = []
with open(all_docs_path, "r") as stream:
    for line in tqdm(stream.readlines()):
        curr_dict = json.loads(line)
        curr_id = curr_dict.get("id")
        curr_text = curr_dict.get("text")
        curr_title = curr_dict.get("title")
        curr_sentences = curr_text.split(". ") # TODO: Better sent frag
        grouped_sents = [
            curr_sentences[itr:itr+offset] \
                for itr in range(0, len(curr_sentences), offset)
        ]
        for group in grouped_sents:
            curr_context = ". ".join(group)
            random_sampled_ans = sample(group, min(len(group), 1))
            random_sampled_ans = ". ".join(random_sampled_ans)

            start_idx = curr_text.find(curr_context)
            end_idx = start_idx + len(curr_context)
            formatted_sent = {
                "_id": gen_sent_idx,
                "title": "",
                "text": copy.deepcopy(t5_qgen_schema).\
                    format(random_sampled_ans, curr_context),
                "metadata": {
                    "src_doc_id": curr_id,
                    "start_id": start_idx,
                    "end_id": end_idx,
                    "_id": gen_sent_idx,
                    "title": curr_title,
                    "text": copy.deepcopy(t5_qgen_schema).\
                        format(random_sampled_ans, curr_context),
                },
            }
            sentences.append(formatted_sent)
            gen_sent_idx += 1

print(f"Total generated sentences: {len(sentences)}")
with open(generated_corpus_path, "w") as stream:
    for sent in sentences:
        stream.write(json.dumps(sent)+"\n")
