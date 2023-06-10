from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForTokenClassification, AutoTokenizer, AutoModel, AutoModelForTokenClassification
import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import uvicorn
from utils.stopwords import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List

app = FastAPI()

@app.get("/")
def read_root():
	return {"message": "Hello World!"}

class TextData(BaseModel):
    text: str

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer2 = AutoTokenizer.from_pretrained('bert-base-uncased')
model2 = AutoModel.from_pretrained('bert-base-uncased')

stop_words = stopwords
# print(stop_words)

class TextData2(BaseModel):
    input: str

model = AutoModelForTokenClassification.from_pretrained('./ner/models/')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('tokenizer_path')
id2label = model.config.id2label  # mapping from label IDs to label names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def split_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def save_bio_keywords(tokens, tags) -> dict:
    # assert len(tokens) == len(tags), "Length of tokens and predicted tags should be the same."
    tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]

    keyword_dict = {}
    current_entity_tokens = []
    current_entity_label = None

    for i, tag in enumerate(tags):
        token = tokens[i]
        if token.startswith('##'):
            if current_entity_tokens:
                token = token[2:]  # remove '##'
                current_entity_tokens[-1] += token  # merge with the previous token
            continue

        if tag.startswith('B'):
            # save previous entity
            if current_entity_label is not None:
                keyword_dict[current_entity_label] = keyword_dict.get(current_entity_label, []) + [' '.join(current_entity_tokens)]
            
            # start new entity
            current_entity_label = tag.split('-')[1]
            current_entity_tokens = [token]
        elif tag.startswith('I') and current_entity_label == tag.split('-')[1]:
            # continue the entity
            if current_entity_label is not None:
                current_entity_tokens.append(token)
        else:
            # save the previous entity if it exists
            if current_entity_label is not None:
                keyword_dict[current_entity_label] = keyword_dict.get(current_entity_label, []) + [' '.join(current_entity_tokens)]
            
            # reset the current entity
            current_entity_tokens = []
            current_entity_label = None

    # add the last entity if it exists
    if current_entity_label is not None:
        keyword_dict[current_entity_label] = keyword_dict.get(current_entity_label, []) + [' '.join(current_entity_tokens)]

    return keyword_dict

def process_keywords_list(keywords_list):
    result = {}

    for item in keywords_list:
        for key, value in item.items():
            if key not in result:
                result[key] = []
            result[key].extend(value)

    return result


@app.post("/ner_inference")
def ner_inference_batch(sen: TextData2):
    word_level_predictions_list = []
    str_rep_list = []
    keywords_list = []

    sen = split_sentences(sen.input)

    for sentence in sen:
        # sentence = sentence.input
        inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

        # move to gpu
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        # forward pass
        outputs = model(ids, mask)
        logits = outputs[0]

        active_logits = logits.view(-1, model.config.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        # print(tokens)
        token_predictions = [id2label[i] for i in flattened_predictions.numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        word_level_predictions = []
        for pair in wp_preds:
            if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
                # skip prediction
                continue
            else:
                word_level_predictions.append(pair[1])

        keywords = save_bio_keywords(tokens, word_level_predictions)
        keywords_list.append(keywords)

        # we join tokens, if they are not special ones
        str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")

        word_level_predictions_list.append(word_level_predictions)
        str_rep_list.append(str_rep)
    
    keywords = process_keywords_list(keywords_list)

    filtered_keywords = {}
    for key, value in keywords.items():
        filtered_values = [word for word in value if word not in stopwords]
        filtered_keywords[key] = filtered_values


    return {"str_rep": str_rep_list,
            # "keywords_list": keywords_list,
            "keywords": filtered_keywords,
            "word_level_predictions": word_level_predictions_list,
            }


sentence_model = SentenceTransformer('all-mpnet-base-v2')
kw_model = KeyBERT(model=sentence_model)

@app.post("/keyword_extraction")
def perform_keyword_extraction(data: TextData2):

    keywords_weights = kw_model.extract_keywords(data.input, 
                                                 keyphrase_ngram_range=(1, 2),  # 추출할 키워드의 n-gram 범위
                                                 diversity=0.9,  # 추출된 키워드의 중복을 허용하는 정도 (1이면 중복 허용 안함)
                                                 stop_words=stop_words,
                                                 top_n=5,  # 추출할 키워드의 개수
                                                 )
    
    keywords = [k_w[0] for k_w in keywords_weights]
    weights = [k_w[1] for k_w in keywords_weights]
    return {"keywords": keywords, "weights": weights}


@app.get("/similarity")
def perform_similarity(keyword1: str, keyword2: str):

    # 두 키워드 각각에 대해 BERT를 이용해 문장 임베딩 수행
    inputs1 = tokenizer2(keyword1, return_tensors='pt', padding=True, truncation=True)
    inputs2 = tokenizer2(keyword2, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs1 = model2(**inputs1)
        outputs2 = model2(**inputs2)

    # 문장 임베딩 결과의 평균을 취함
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    # 두 임베딩 벡터 간의 코사인 유사도 계산
    similarity = cosine_similarity(embeddings1, embeddings2)

    return {'similarity': similarity.item()}

# @app.post("/similarity")
# def perform_similarity(keyword1: str, keyword2: str):
#     app_similarity(keyword1, keyword2)



if __name__ == '__main__':
	uvicorn.run(app, host="0.0.0.0", port=8000) 
