1. Install elasticsearch (https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)
2. Run elasticsearch as a service (https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started-install.html)
3. Run elasticsearch health check to see if it's running:
```
curl http://localhost:9200/_cat/health?v=true
```
4. You can also run the following command to get basic info of your elasticsearch cluster and node:
```
curl http://localhost:9200
```
5. If you are using elasticsearch with Python, install Python client for elasticsearch for easier manipulation of es in Python:
```
pip install elasticsearch
```
(Or whatever installing command that works for you. For me, I use pyenv virtual environment and run this command after starting up the virtual environment)
