import json
import os
import pdb

def _load_doc_data_rc(doc_filepath):
    # doc_filepath = os.path.dirname(filepath)
    with open(doc_filepath, encoding="utf-8") as f:
        # Corpus
        data = json.load(f)["doc_data"]
    
    return data

def load_query(filepath):
    with open(filepath, encoding="utf-8") as f:
        dial_data = json.load(f)["dial_data"]

    return dial_data

corpus_path = "/home/ubuntu/ANLP/relevance-IR/data/doc2dial_doc.json"
data = _load_doc_data_rc(corpus_path)

# "dial_data": {
#         "dmv": {
#             "Top 5 DMV Mistakes and How to Avoid Them#3_0": [
#                 {
#                     "dial_id": "dea7174409afbfe0af0ace21e7f318ae",
#                     "doc_id": "Top 5 DMV Mistakes and How to Avoid Them#3_0",
#                     "domain": "dmv",
#                     "turns": [
#                         {
#                             "turn_id": 1,
#                             "role": "user",
#                             "da": "query_condition",
#                             "references": [
#                                 {
#                                     "sp_id": "23",
#                                     "label": "precondition"
#                                 }
#                             ],
#                             "utterance": "My insurance ended so what should i do"
#                         },
#                         {
#                             "turn_id": 2,
#                             "role": "agent",
#                             "da": "respond_solution",
#                             "references": [
#                                 {
#                                     "sp_id": "24",
#                                     "label": "solution"
#                                 },
#                                 {
#                                     "sp_id": "25",
#                                     "label": "solution"
#                                 },
#                                 {
#                                     "sp_id": "26",
#                                     "label": "solution"
#                                 }
#                             ],
#                             "utterance": "You will need to get insurance or we will suspend your registration and license"
                       
    

## Corpus
with open("/home/ubuntu/ANLP/relevance-IR/data/beir-format/corpus-without-title.jsonl", "w") as f:
    for key in data.keys():
        for k in data[key].keys():
            doc_id = data[key][k]["doc_id"]
            # title = data[key][k]["title"]
            title = ""
            doc_text = data[key][k]["doc_text"]
            example = {"id": doc_id, "title": title, "text": doc_text}
            json.dump(example, f)
            f.write("\n")

# def writeFile(file_path, data, n, reverse):
    
#     # if(reverse):
#     #     file_a = file_path+"train-queries-reverse-"+str(n)+".jsonl"
#     #     file_b = file_path+"train-qrels-reverse-"+str(n)+".tsv"

#     # else:
#     #     file_a = file_path+"train-queries-"+str(n)+".jsonl"
#     #     file_b = file_path+"train-qrels-"+str(n)+".tsv"
    
#     if(reverse):
#         file_a = file_path+"test-queries-reverse-"+str(n)+".jsonl"
#         file_b = file_path+"test-qrels-reverse-"+str(n)+".tsv"

#     else:
#         file_a = file_path+"test-queries-"+str(n)+".jsonl"
#         file_b = file_path+"test-qrels-"+str(n)+".tsv"
    

#     with open(file_a, "w") as f:
#         with open(file_b, "w") as fw:

#             query_id = 0
#             for domain, d_doc_dials in data.items():
#                 for doc_id, dials in d_doc_dials.items():
#                     # doc = doc_data[domain][doc_id]
#                     for dial in dials:
#                         doc_id = dial["doc_id"]
#                         all_prev_utterances = []
#                         for idx, turn in enumerate(dial["turns"]):
#                             all_prev_utterances.append(
#                                 "{}: {}".format(turn["role"], turn["utterance"])
#                             )
                            
#                         query_id += 1

#                         if(reverse):
#                             temp = " ".join(all_prev_utterances[-n:])
#                         else:
#                             temp = " ".join(all_prev_utterances[:n])

#                         example = {"id": query_id, "text": temp}
#                         json.dump(example, f)
#                         f.write("\n")

#                         example = str(query_id)+"\t"+str(doc_id)+"\t"+"1"
#                         fw.write(example+"\n")
                    
#             # for key in data.keys():
#             #     for k in data[key].keys():
#             #         pdb.set_trace()
#             #         doc_id = data[key][k]["doc_id"]
#             #         query_id += 1
#             #         turns = data[key][k]["turns"]
                    
#             #         temp = ""
#             #         for i, turn in enumerate(turns):
#             #             role = turn["role"]
#             #             utterance = turn["enumerate"]
#             #             temp = "<"+role+">" + " " + utterance + " " + temp
                        
#             #             if(i==n):
#             #                 example = {"id": query_id, "text": temp}
#             #                 json.dump(example, f)
#             #                 f.write("\n")

#             #                 example = str(query_id)+"\t"+str(doc_id)+"\t"+"1"
#             #                 fw.write(example+"\n")
#             #                 break

# # query_path = "/home/ubuntu/ANLP/relevance-IR/data/doc2dial_dial_train.json"
# query_path = "/home/ubuntu/ANLP/relevance-IR/data/doc2dial_dial_validation.json"
# query_data = load_query(query_path)
                
# for i in range(1, 6):
#     writeFile("/home/ubuntu/ANLP/relevance-IR/data/beir-format/test/", query_data, i, reverse=False)

# # # Queries
# # with open("/home/ubuntu/ANLP/relevance-IR/data/queries.jsonl", "w") as f:
# #     for key in data.keys():
# #         for k in data[key].keys():
# #             doc_id = data[key][k]["doc_id"]
# #             turns = data[key][k]["turns"]
            
# #             temp = ""
# #             for i, turn in enumerate(turns):
# #                 role = turn["role"]
# #                 utterance = turn["enumerate"]

# #             title = data[key][k]["title"]
# #             doc_text = data[key][k]["doc_text"]
# #             example = {"id": doc_id, "title": title, "text": doc_text}
# #             json.dump(example, f)
# #             f.write("\n")

# # features = {
# #                     "domain": datasets.Value("string"),
# #                     "doc_id": datasets.Value("string"),
# #                     "title": datasets.Value("string"),
# #                     "doc_text": datasets.Value("string"),
# #                     "spans": [
# #                         {
# #                             "id_sp": datasets.Value("string"),
# #                             "tag": datasets.Value("string"),
# #                             "start_sp": datasets.Value("int32"),
# #                             "end_sp": datasets.Value("int32"),
# #                             "text_sp": datasets.Value("string"),
# #                             "title": datasets.Value("string"),
# #                             "parent_titles": datasets.Value("string"),
# #                             "id_sec": datasets.Value("string"),
# #                             "start_sec": datasets.Value("int32"),
# #                             "text_sec": datasets.Value("string"),
# #                             "end_sec": datasets.Value("int32"),
# #                         }
# #                     ],
# #                     "doc_html_ts": datasets.Value("string"),
# #                     "doc_html_raw": datasets.Value("string"),
# #                 }
# #             )
# #             }