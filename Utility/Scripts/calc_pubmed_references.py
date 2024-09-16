import os
import json
from collections import defaultdict

# fromdir = '/Users/xinzhaoli/Documents/Research/pubmed_data/unzipped_exp'
# fromdir = '/Volumes/Seagate Portable Drive/pubmed_data/unzipped'
fromdir = '/Users/xinzhaoli/Documents/Research/pubmed_data/json_raw_citedby'
todir = '/Users/xinzhaoli/Documents/Research/pubmed_data/json_num_citedby_exp'
if not os.path.exists(todir):
    os.mkdir(todir)

total_json = defaultdict(int)
i = 0
for filename in os.listdir(fromdir):
    i += 1
    print('Calculating file # ' + str(i))
    path = os.path.join(fromdir, filename)
    with open(path, 'r') as infile:
        file_json = json.load(infile)

    for k, v in file_json.items():
        total_json[k] += len(v)


    outname = os.path.splitext(os.path.basename(filename))[0] + '-raw-citedby.json'
    outname = 'pumbed-num-citedby-total.json'
    outpath = os.path.join(todir, outname)
    with open(outpath, 'w') as outfile:
        json.dump(total_json, outfile)
