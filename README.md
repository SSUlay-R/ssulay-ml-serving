# ssulay-ml-serving

## NER

### 로컬 실행

uvicorn main:app --reload

API endpoint

- http://<ip_address>:8000/ner_inference
- http://<ip_address>:8000/keyword_extraction
- http://<ip_address>:8000/similarity

API docs

- http://<ip_address>:8000/docs
- http://<ip_address>:8000/redoc

### 서버 (데몬)

- sudo vim /etc/nginx/sites-available/ssulay-nlp
- sudo ln -s /etc/nginx/sites-available/ssulay-nlp /etc/nginx/sites-enabled/

- source venv/bin/activate
- python3 -m gunicorn -k uvicorn.workers.UvicornWorker main:app --daemon
- python3 -m gunicorn -k uvicorn.workers.UvicornWorker main:app --daemon --access-logfile ./gunicorn-access.log

- sudo systemctl restart nginx.service
