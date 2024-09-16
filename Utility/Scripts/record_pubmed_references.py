import os
import xml.etree.ElementTree as ET
import json
from collections import defaultdict

# fromdir = '/Users/xinzhaoli/Documents/Research/pubmed_data/unzipped_exp'
fromdir = '/Volumes/Seagate Portable Drive/pubmed_data/unzipped'
todir = '/Users/xinzhaoli/Documents/Research/pubmed_data/json_raw_citedby'
if not os.path.exists(todir):
    os.mkdir(todir)
i = 0
for filename in os.listdir(fromdir):
    i += 1
    print('Recording file # ' + str(i))
    path = os.path.join(fromdir, filename)
    tree = ET.parse(path)
    root = tree.getroot()
    record_json = defaultdict(list)
    for article in root:
        citing_pmid = article.find('MedlineCitation').find('PMID').text # currently article being parsed
        references = article.iter('Reference')
        for ref in references:
            article_ids = ref.iter('ArticleId')
            for aid in article_ids:
                if aid.attrib['IdType'] == 'pubmed':
                    record_json[aid.text].append(citing_pmid) # collect every pmid cited by current article

    outname = os.path.splitext(os.path.basename(filename))[0] + '-raw-citedby.json'
    outpath = os.path.join(todir, outname)
    with open(outpath, 'w') as outfile:
        json.dump(record_json, outfile)
