from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

@app.get("/similarity")
def perform_similarity(keyword1: str, keyword2: str):
    # 두 키워드 각각에 대해 BERT를 이용해 문장 임베딩 수행
    inputs1 = tokenizer(keyword1, return_tensors='pt', padding=True, truncation=True)
    inputs2 = tokenizer(keyword2, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # 문장 임베딩 결과의 평균을 취함
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    # 두 임베딩 벡터 간의 코사인 유사도 계산
    similarity = cosine_similarity(embeddings1, embeddings2)

    return {'similarity': similarity.item()}