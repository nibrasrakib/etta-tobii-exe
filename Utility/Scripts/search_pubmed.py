from elasticsearch import Elasticsearch
import time

es = Elasticsearch()

query_body = {
    "query": {
        "match": {
            "title": "central heating"
        }
    }
}

start = time.time()
response = es.search(body=query_body, index="pubmed", size=10)
end = time.time()
print("time: " + str(end - start))
print(response)
