from fastapi import FastAPI
from pydantic import BaseModel
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

app = FastAPI()

# MODEL_NAME = 'all-mpnet-base-v2'
# sentence_model = SentenceTransformer(MODEL_NAME)
# kw_model = KeyBERT(model=sentence_model)

class DocData(BaseModel):
    text: str

stopwords = ['i', 'me', 'my', 'myself', 'im', 'm', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 'dont',
			'should', 'now', '']


@app.post("/keyword_extraction")
def app(data: DocData):
    MODEL_NAME = 'all-mpnet-base-v2'
    sentence_model = SentenceTransformer(MODEL_NAME)
    kw_model = KeyBERT(model=sentence_model)

    N_GRAM_RANGE = (1, 2)
    DIVERSITY = 0.9
    keywords_weights = kw_model.extract_keywords(data.text, 
                                                 keyphrase_ngram_range=N_GRAM_RANGE,  # 추출할 키워드의 n-gram 범위
                                                 diversity=DIVERSITY,  # 추출된 키워드의 중복을 허용하는 정도 (1이면 중복 허용 안함)
                                                 stop_words=stopwords,
                                                 top_n=5,  # 추출할 키워드의 개수
                                                 )
    
    keywords = [k_w[0] for k_w in keywords_weights]
    weights = [k_w[1] for k_w in keywords_weights]
    return {"keywords": keywords, "weights": weights}
