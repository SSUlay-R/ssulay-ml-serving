import requests

# 서버 URL
BASE_URL = '127.0.0.1:8000'
# BASE_URL = '219.255.158.173:8000'
url = f"http://{BASE_URL}/ner_inference"

# 인풋 데이터
data = {
    "text": "I like crossfit and scuba diving, and I also like to play the guitar."
}
# input_text = 'I like crossfit and scuba diving'

# POST 요청 보내기
response = requests.post(url, json=data)

# 응답 확인
if response.status_code == 200:
    result = response.json()
    entities = result['entities']
    pred_tags = result['pred_tags']
    print(result)
    # print('entities:', entities)
    # print('Predicted Tags:', pred_tags)
else:
    print('Error:', response.text)