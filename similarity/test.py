import requests

# 서버 URL
BASE_URL = '127.0.0.1:8000'
url = f"http://{BASE_URL}/similarity"

# # 인풋 데이터
params = {
    "keyword1": "latte",
    "keyword2": "coffee",
}

# POST 요청 보내기
response = requests.get(url, params=params)

# 응답 확인
if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print('Error:', response.text)