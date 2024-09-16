from elasticsearch import Elasticsearch, helpers
import os
import json
import time

patch_size = 3000
es = Elasticsearch()

basedir = '/Users/xinzhaoli/Documents/Research/pubmed_data/json2_exp'
cite_file = open('/Users/xinzhaoli/Documents/Research/pubmed_data/json_num_citedby/pumbed-num-citedby-total.json', 'r')
cite_json = json.load(cite_file)

def update_json_bulk():
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
                    # year = article_json["source"]["pubDate"]["year"]
                    # if year:
                    #     article_json["source"]["pubDate"]["year"] = int(year)
                    if pmid in cite_json:
                        current_patch.append({
                            "_op_type": "update",
                            "_index": "pubmed",
                            "_id": pmid,
                            "doc": {"num_cited_by": cite_json[pmid]}
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

update_json_bulk()
