import requests
import json

# 서버 URL
BASE_URL = '127.0.0.1:8000'
url = f"http://{BASE_URL}/keyword_extraction"

data = {
    "doc": "Hi there! My name is Sarah and I'm a fitness enthusiast who loves to stay active outdoors. I'm an avid runner and cyclist, and I also enjoy practicing yoga and doing crossfit. When I'm not working out, you can find me hiking in the mountains or camping by the lake. I'm always up for an adventure!",
    "top_n": 5,
}

response = requests.post(url, json=data)
print(response)

# if response.status_code == 200:
#     result = response.json()
#     # keywords = result['keywords']
#     # weights = result['weights']
#     print(result)
#     # print('keywords:', keywords)
#     # print('weights:', weights)
# else:
#     print('Error:', response.text)