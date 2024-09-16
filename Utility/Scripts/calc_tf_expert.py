import json
import os
import ast
from collections import Counter
import re

dir = '/Users/xinzhaoli/Documents/Research/expert_data'
inname = 'expert_database.txt'
outname1 = 'expert_tf_full.txt'
outname2 = 'expert_tf_top10percent.txt'

tokenize = re.compile("[^\w\-]+")  # a token is composed of
path = os.path.join(dir, inname)
with open(path, 'r') as f:
    data_string = f.read()
    data_string = str(data_string)
    data = ast.literal_eval(data_string)
    bio_total = '' 
    for name, profile in data.items():
        biography = profile.get('biography', '')
        bio_total += biography.lower()

    terms = tokenize.split(bio_total)
    counter = Counter(terms)
    result1 = counter
    outpath1 = os.path.join(dir, outname1)
    with open(outpath1, 'w') as f1:
        f1.write(str(result1))

    result2 = counter.most_common(int(len(counter) * 0.1))
    outpath2 = os.path.join(dir, outname2)
    with open(outpath2, 'w') as f2:
        f2.write(str(result2))




