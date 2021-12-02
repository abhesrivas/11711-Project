import numpy as np
import pdb

# train_data = np.genfromtxt("/home/ubuntu/Karthik/11711-Project/two-step-reader/data/cross-encoder-data/cross-encoder-train.csv", delimiter="\t", dtype="str", filling_values="<MISSING>")
# test_data = np.genfromtxt("/home/ubuntu/Karthik/11711-Project/two-step-reader/data/cross-encoder-data/cross-encoder-test.csv", delimiter="\t", dtype="str", filling_values="<MISSING>")



examples = []
with open("/home/ubuntu/Karthik/11711-Project/two-step-reader/data/cross-encoder-data/cross-encoder-train.csv", "r") as f:
    for line in f.readlines():
        if(len(line.split("\t")) == 3):
            a = line.split("\t")
            pdb.set_trace()
            examples.append(a)

pdb.set_trace()
train_examples = [example for example in train_data if example[2]=="0"]
test_examples = [example for example in test_data if example[2]=="0"]

pdb.set_trace()