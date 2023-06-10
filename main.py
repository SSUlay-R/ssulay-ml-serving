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

app = FastAPI()

@app.get("/")
def read_root():
	return {"message": "Hello World!"}

class TextData(BaseModel):
    text: str

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer2 = AutoTokenizer.from_pretrained('bert-base-uncased')
model2 = AutoModel.from_pretrained('bert-base-uncased')


class TextData2(BaseModel):
    input: str

model = AutoModelForTokenClassification.from_pretrained('./ner/models/')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('tokenizer_path')
id2label = model.config.id2label  # mapping from label IDs to label names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.post("/ner_inference")
def ner_inference(sen: TextData2):
    sentence = sen.input
    
    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    # forward pass
    outputs = model(ids, mask)
    logits = outputs[0]

    print('>>', logits.shape)  # >> torch.Size([1, 512, 768])
    print('>>', model.config.num_labels)  # >> 17
    active_logits = logits.view(-1, model.config.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [id2label[i] for i in flattened_predictions.numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    word_level_predictions = []
    for pair in wp_preds:
        if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
            # skip prediction
            continue
        else:
            word_level_predictions.append(pair[1])

    # we join tokens, if they are not special ones
    str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")

    return {"str_rep": str_rep,
            "word_level_predictions": word_level_predictions,}

stop_words = stopwords
# print(stop_words)

@app.post("/keyword_extraction")
def perform_keyword_extraction(data: TextData, top_n: int = 5):
    MODEL_NAME = 'all-mpnet-base-v2'
    sentence_model = SentenceTransformer(MODEL_NAME)
    kw_model = KeyBERT(model=sentence_model)

    keywords_weights = kw_model.extract_keywords(data.text, 
                                                 keyphrase_ngram_range=(1, 2),  # 추출할 키워드의 n-gram 범위
                                                 diversity=0.9,  # 추출된 키워드의 중복을 허용하는 정도 (1이면 중복 허용 안함)
                                                 stop_words=stop_words,
                                                 top_n=top_n,  # 추출할 키워드의 개수
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
