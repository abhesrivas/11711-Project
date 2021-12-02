import json

final_dictionary = {}
final_dictionary["question"] = []
final_dictionary["context"] = []
final_dictionary["id"] = []
final_dictionary["title"] = []
final_dictionary["answers"] = []
final_dictionary["domain"] = []

def _get_answers_rc(self, references, spans, doc_text):
    """Obtain the grounding annotation for evaluation of subtask1."""
    if not references:
        return []
    start, end = -1, -1
    ls_sp = []
    for ele in references:
        sp_id = ele["sp_id"]
        start_sp, end_sp = spans[sp_id]["start_sp"], spans[sp_id]["end_sp"]
        if start == -1 or start > start_sp:
            start = start_sp
        if end < end_sp:
            end = end_sp
        ls_sp.append(doc_text[start_sp:end_sp])
    answer = {
        "text": doc_text[start:end],
        "answer_start": start,
        # "spans": ls_sp
    }
    return [answer]

output = []
with open('formatted-gen-3-queries.jsonl','r') as f:
    for line in f.readlines():
        try:
            current_json = json.loads(line)
            question = current_json["text"].split("?")[0]
            context = current_json["metadata"]["text"].split("context_token")[1]
            #print(len(current_json["metadata"]["text"].split("context_token")))
            #print(current_json["metadata"]["text"].split("context_token"))

            answer_span = current_json["metadata"]["text"].split("context_token")[0].split("answer_token")[1]
            start_id = current_json["metadata"]["start_id"]
            _id = current_json["_id"]
            title = current_json["metadata"]["title"]
            '''final_dictionary["question"].append(question)
            final_dictionary["context"].append(context)
            final_dictionary["id"].append(_id)
            final_dictionary["title"].append(title)
            final_dictionary["answers"].append(answer)
            final_dictionary["domain"].append("public_forum")'''

            current_json = {
                'answers': {
                    'answer_start': [start_id], 
                    'text': [answer_span]
                },
                'context': context, 
                'domain': "public_forum", 
                'id': _id , 
                'question': question, 
                'title': title 
            }
        
        except:
            continue

        output.append(current_json) 

    generated_q_counter = 0
    with open('output.jsonl', 'w') as outfile:
        for entry in output:
            outfile.write(json.dumps(entry)+"\n")
            generated_q_counter += 1
        #print(current_json["metadata"]["text"])

    print(f"generated questions: {generated_q_counter}")   

assert len(final_dictionary["question"])==len(final_dictionary["context"])==len(final_dictionary["id"])==len(final_dictionary["answers"])==len(final_dictionary["domain"])    
'''with open('final_output.json','w') as f:
    json.dump(final_dictionary,f, indent=4)'''
    