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
train_dataset_path = '/home/ubuntu/Karthik/11711-Project/two-step-reader/data/cross-encoder-data/cross-encoder-train-balanced.csv'
test_dataset_path = '/home/ubuntu/Karthik/11711-Project/two-step-reader/data/cross-encoder-data/cross-encoder-test-balanced.csv'
eval_dataset_path = '/home/ubuntu/Karthik/11711-Project/two-step-reader/data/cross-encoder-data/cross-encoder-test-balanced-1000.csv'

#Define our Cross-Encoder
train_batch_size = 20
num_epochs = 4
model_save_path = '../output/cross-encoder-demo-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#We use distilroberta-base as base model and set num_labels=1, which predicts a continous score between 0 and 1
model = CrossEncoder('distilroberta-base', num_labels=1)

# Read STSb dataset
logger.info("Read CE train dataset")

train_samples = []
test_samples = []
eval_samples = []

with open(train_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE, fieldnames=['sentence1', 'sentence2', 'score'])
    for row in reader:
        score = float(row['score'])  # Normalize score to range 0 ... 1
        train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

with open(test_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE, fieldnames=['sentence1', 'sentence2', 'score'])
    for row in reader:
        score = float(row['score'])  # Normalize score to range 0 ... 1
        test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

with open(eval_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE, fieldnames=['sentence1', 'sentence2', 'score'])
    for row in reader:
        score = float(row['score'])  # Normalize score to range 0 ... 1
        eval_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
evaluator = CEBinaryAccuracyEvaluator.from_input_examples(eval_samples, name='dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

##### Load model and eval on test set
model = CrossEncoder(model_save_path)

# corr_evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='correlation-test')
# corr_evaluator(model)

binary_evaluator = CEBinaryAccuracyEvaluator.from_input_examples(test_samples, name='binary-test')
binary_evaluator(model)