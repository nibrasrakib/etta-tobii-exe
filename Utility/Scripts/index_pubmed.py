from elasticsearch import Elasticsearch, helpers
import os
import json
import time

basedir = '/Users/xinzhaoli/Documents/Research/pubmed_data/json2_exp'
patch_size = 3000
es = Elasticsearch()

def load_json_bulk():
    current_patch = []
    i = 0
    for filename in os.listdir(basedir):
        if filename.endswith('.json'):
            start = time.time()
            read_path = os.path.join(basedir, filename)
            with open(read_path) as f:
                data = json.load(f)
                for pmid, article_json in data.items():
                    i += 1
                    article_json['num_cited_by'] = 0
                    # year = article_json["source"]["pubDate"]["year"]
                    # if year:
                    #     article_json["source"]["pubDate"]["year"] = int(year)
                    current_patch.append({
                        "_index": "pubmed",
                        "_id": pmid,
                        "_source": article_json
                    })
                    if i == patch_size:
                        try:
                            response = helpers.bulk(es, current_patch)
                            # print ("\nRESPONSE:", response)
                        except Exception as e:
                            print("\nERROR:", e)
                        current_patch = []
                        i = 0
                
            print(filename + " finished processing")
            end = time.time()
            print("time: " + str(end - start))
            print(i)
    if current_patch:
        try:
            response = helpers.bulk(es, current_patch)
            # print ("\nRESPONSE:", response)
        except Exception as e:
            print("\nERROR:", e)
        current_patch = []

load_json_bulk()
