from elasticsearch import Elasticsearch, helpers
import json
import os
import ast

dir = '/Users/xinzhaoli/Documents/Research/expert_data'
inname = 'expert_database.txt'
es = Elasticsearch()
patch_size = 300

path = os.path.join(dir, inname)
with open(path, 'r') as f:
    data_string = f.read()
    data_string = str(data_string)
    data = ast.literal_eval(data_string)
    current_patch = []
    i = 0
    for name, profile in data.items():
        # print(profile)
        education_json = []
        for institution, degree in profile.get('education', {}).items():
            education_json.append({'institution': institution, 'degree': degree})
        expert_json = {
                'name': name,
                'affiliations': profile.get('affiliations', []),
                'expert_title': profile.get('title', []),
                'expert_title_description': profile.get('title_description', []),
                'biography': profile.get('biography', ''),
                'industry_expertise': profile.get('industry_expertise', []),
                'areas_expertise': profile.get('areas_expertise', []),
                'education': education_json,
                'url': profile['url'],
                }
        current_patch.append({
            "_index": "experts",
            "_source": expert_json
        })
        if i == patch_size:
            try:
                response = helpers.bulk(es, current_patch)
                print ("\nRESPONSE:", response)
            except Exception as e:
                print("\nERROR:", e)
            current_patch = []
            i = 0

    if current_patch:
        try:
            response = helpers.bulk(es, current_patch)
            print ("\nRESPONSE:", response)
        except Exception as e:
            print("\nERROR:", e)
        current_patch = []



