"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator, CEBinaryAccuracyEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import pdb

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
test_dataset_path = '/home/ubuntu/Karthik/11711-Project/two-step-reader/data/cross-encoder-data/cross-encoder-test-balanced.csv'

test_samples = []

with open(test_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE, fieldnames=['sentence1', 'sentence2', 'score'])
    for row in reader:
        score = float(row['score'])  # Normalize score to range 0 ... 1
        test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

model_save_path = "/home/ubuntu/Karthik/11711-Project/two-step-reader/output/cross-encoder-demo-2021-12-01_07-27-20"

##### Load model and eval on test set
model = CrossEncoder(model_save_path)

# corr_evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='correlation-test')
# corr_evaluator(model)

binary_evaluator = CEBinaryAccuracyEvaluator.from_input_examples(test_samples, name='binary-test')
binary_evaluator(model)