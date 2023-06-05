import requests

# 서버 URL
BASE_URL = '127.0.0.1:8000'
url = f"http://{BASE_URL}/similarity"

params = {
    "keyword1": "latte",
    "keyword2": "coffee",
}

response = requests.get(url, params=params)

if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print('Error:', response.text)