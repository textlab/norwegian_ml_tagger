import sys
import torch
import os
import transformers
import numpy
import copy
import json
from transformers import BertTokenizerFast
from transformers import EncoderDecoderModel
from transformers import BertModel
from transformers import BertConfig
from transformers import BertLMHeadModel
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from functools import cmp_to_key


os.environ["TOKENIZERS_PARALLELISM"]="false"
enc_max_length=512
int_classification_device=-1  # -1 for cpu or gpu id
int_tokenization_device=-1    # -1 for cpu or gpu id 

label_text_file = "data/label_list.txt"
classes_file= "data/labels_classifier.txt"
with open("data/labels_order.json","r") as f:
    label_order = json.load(f)

if int_classification_device == -1:
    classification_device="cpu"
else:
    classification_device="cuda:" + str(int_classification_device)

if int_tokenization_device == -1:
    tokenization_device="cpu"
else:
    tokenization_device="cuda:" + str(int_tokenization_device)


tag_enc_tokenizer = BertTokenizerFast.from_pretrained('NbAiLab/nb-bert-base')

#torch.cuda.set_device(classification_device)

classification_model = AutoModelForTokenClassification.from_pretrained("./models/classifier/")
classification_model.to(classification_device)
classification_model.eval()
#classifier_pipeline = pipeline('ner', model = classification_model, tokenizer = tag_enc_tokenizer, device = int_classification_device)

#torch.cuda.set_device(tokenization_device)

tokenization_model = AutoModelForTokenClassification.from_pretrained("./models/tokenization/")
tokenization_model.to(tokenization_device)
tokenization_model.eval()
tokenization_pipeline = pipeline('ner', model = tokenization_model, tokenizer = tag_enc_tokenizer, device = int_tokenization_device)

torch.no_grad()


# Get list of labels
label_list=None
class_list=None
class_to_label={}
with open(label_text_file, "r") as f:
    label_list= [i for i in f.read().split("\n") if i!=""]
classes=[]
with open(classes_file,"r") as f:
    class_list = [i for i in f.read().split("\n") if i!=""]
for c in class_list:
    classes=set()
    for i in range(len(c)):
        if c[i]=="1":
            classes.add(label_list[i])
    class_to_label[c] = classes

def compare_label(t1,t2):
    global label_order
    val1=-1
    val2=-1
    key1 = t1 + " " + t2
    key2 = t2 + " " + t1
    if key1 in label_order:
        val1=label_order[key1]
    if key2 in label_order:
        val2=label_order[key2]
    if val1>val2:
        return -1
    return 1
    
cmp_key = cmp_to_key(compare_label)   


def get_classes(text, topmost=10):
    global classification_model
    global tag_enc_tokenizer
    global classification_device
    global class_to_label

    inputs = tag_enc_tokenizer(text, return_tensors="pt" )
    inputs.to(classification_device)

    with torch.no_grad():
        logits = classification_model(**inputs).logits
        
#    predictions = torch.argmax(logits, dim=2)
    
    predictions=torch.topk(logits.flatten(start_dim=2, end_dim=2), topmost).indices
    predictions=torch.rot90(predictions[0],1,[0,1])
    predictions=torch.flip(predictions,[0])

    classes = [ [class_to_label[classification_model.config.id2label[t.item()]] for t in i] for i in predictions ]
    return classes

def tag_sentence(text,topmost=10):

    global tokenization_pipeline
#    global classifier_pipeline
    global labels
    
    classes=get_classes(text,topmost)
    token_merges = tokenization_pipeline(text, ignore_labels=[] )
#    classes = classifier_pipeline(text, ignore_labels=[] )

    all_possibilities=[]
    for classs in classes:
        token_list=[]

        merge_next_token=False
        for (t, c)  in zip(token_merges, classs[1:-1]):
            if len(t["word"])>1 and t["word"][0:2]=="##":
                token_list[len(token_list)-1][0]+=t["word"][2:]
            elif merge_next_token:
                token_list[len(token_list)-1][0]+=t["word"]
                merge_next_token=False
            else:
                token_list.append([t["word"]," ".join(sorted(list(c),key=cmp_key))])
            if t["entity"]=="P":
                merge_next_token=True
        
        all_possibilities.append([token_list,0])
    return all_possibilities


def main():

    done=""
    try:
        f=open(sys.argv[1],"r")
        line=f.readline()
        return_list=[]
        while line:
            line=line.strip()
            done+=line+"\n"
            return_list.append(tag_sentence(line,1)[0][0])
            line=f.readline()
        f.close()
        print(json.dumps(return_list))
    except Exception as e:
        eprint(done)
        eprint(e)

if __name__ == "__main__":
    main()
    #print(tag_sentence("Dansk ungdom tyr i dag i større grad enn før til engelsk når de møter svensker og nordmenn.", 10)[0][0])

