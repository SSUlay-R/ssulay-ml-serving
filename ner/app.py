from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForTokenClassification
import numpy as np

app = FastAPI()

class TextData(BaseModel):
    text: str

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased')
model.classifier = torch.nn.Linear(768, 15)

model_dict = torch.load("./models/pytorch_model.bin", map_location=device)
model.load_state_dict(model_dict)
model.to(device)
model.eval()

entity_tags = ['FOOD', 'TECH', 'ENT', 'CHARITY', 'ART', 'OUT', 'MUSIC']

# 각 tag를 고유 id(정수값)로 매핑
unique_tags = set()
for tag in entity_tags:
    unique_tags.add('B-' + tag)
    unique_tags.add('I-' + tag)

unique_tags.add('O')
# unique_tags.add('[PAD]')

tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

def encode_data(tokens_docs, tag_docs, max_seq_length):
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    encoded_inputs = []
    encoded_tags = []
    
    for tokens in tokens_docs:
        # Convert tokens to input encoding
        encoded_input = tokenizer.encode_plus(tokens,
                                              truncation=True,
                                              padding='max_length',
                                              max_length=max_seq_length,
                                              return_attention_mask=True,
                                              return_tensors='pt')
        encoded_inputs.append(encoded_input)
    
    for tags in tag_docs:
        # Convert tag labels to input encoding
        encoded_tag = [tag2id[tag] for tag in tags]
        
        # Truncate or pad the tag sequence
        if len(encoded_tag) > max_seq_length:
            encoded_tag = encoded_tag[:max_seq_length]
        else:
            encoded_tag += [0] * (max_seq_length - len(encoded_tag))
        
        encoded_tags.append(encoded_tag)
    
    # Convert encoded inputs to tensors
    input_ids = torch.cat([encoded_input['input_ids'] for encoded_input in encoded_inputs], dim=0)
    attention_masks = torch.cat([encoded_input['attention_mask'] for encoded_input in encoded_inputs], dim=0)
    
    # Convert encoded tags to tensors
    tags_ids = torch.tensor(encoded_tags)
    
    return input_ids, attention_masks, tags_ids

# # 모델 초기화 및 예측 함수 정의
# def predict(data):
#     # 예측 로직 구현
#     ...
@app.post("/ner_inference")
def perform_ner_inference(data: TextData):
    # model.eval()

    # Tokenize input text
    tokens = data.text.split()
    input_ids, attention_masks, tags_ids = encode_data([tokens], [[]], len(tokens))

    # Move tensors to the appropriate device
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    tags_ids = tags_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=tags_ids
        )

    logits = outputs['logits']
    logits = logits.detach().cpu().numpy()
    label_ids = tags_ids.cpu().numpy()

    predictions = [list(p) for p in np.argmax(logits, axis=2)]
    pred_tags = [id2tag[p_i] for p in predictions for p_i in p]

    flag = False
    start, end = 0, 0
    text_vector = []
    entity_tags = []

    for token, tag in zip(tokens, pred_tags):
        if tag == 'O':
            if flag:
                end = len(entity_tags)
                entity = ' '.join(tokens[start:end])
                text_vector.append(entity)
                flag = False
            entity_tags.append(tag)
        else:
            if not flag:
                start = len(entity_tags)
                flag = True
            entity_tags.append(tag)

    # Check if an entity is still open at the end
    if flag:
        entity = ' '.join(tokens[start:])
        text_vector.append(entity)

    return {"entities": text_vector, "pred_tags": pred_tags}

# @app.post("/predict")
# def run_prediction(input_data: TextData):
#     data = input_data.data
#     prediction = predict(data)
#     return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)