import os
import json
from collections import defaultdict

# fromdir = '/Users/xinzhaoli/Documents/Research/pubmed_data/unzipped_exp'
# fromdir = '/Volumes/Seagate Portable Drive/pubmed_data/unzipped'
fromdir = '/Users/xinzhaoli/Documents/Research/pubmed_data/json2_exp'
todir = '/Users/xinzhaoli/Documents/Research/pubmed_data/json2+'
if not os.path.exists(todir):
    os.mkdir(todir)


cite_file = open('/Users/xinzhaoli/Documents/Research/pubmed_data/json_num_citedby/pumbed-num-citedby-total.json', 'r')
cite_json = json.load(cite_file)
i = 0
for filename in os.listdir(fromdir):
    if not filename.endswith('.json'):
        continue
    i += 1
    print('Modifying file # ' + str(i))
    path = os.path.join(fromdir, filename)
    with open(path, 'r') as infile:
        file_json = json.load(infile)

        for k, v in file_json.items():
            v['num_cited_by'] = cite_json[k]

    outname = filename
    outpath = os.path.join(todir, outname)
    with open(outpath, 'w') as outfile:
        json.dump(file_json, outfile)
