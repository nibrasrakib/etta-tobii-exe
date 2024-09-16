import xml.etree.ElementTree as ET
import json
import os
print('start parsing...')
# fromdir = '/Users/xinzhaoli/Documents/Research/pubmed_data/unzipped_exp'
fromdir = '/Volumes/Seagate Portable Drive/pubmed_data/unzipped'
todir = '/Users/xinzhaoli/Documents/Research/pubmed_data/json2'
if not os.path.exists(todir):
    os.mkdir(todir)

i = 0
for filename in os.listdir(fromdir):
    i += 1
    print('Parsing file # ' + str(i))
    path = os.path.join(fromdir, filename)
    tree = ET.parse(path)
    root = tree.getroot()
    total_json = {}
    for article in root:
        title = article.find('MedlineCitation').find('Article').find('ArticleTitle').text
        abstract = list(article.iter('AbstractText'))
        abstract_text = '' if abstract is not None else None
        for abst in abstract:
            if abst.text:
                abstract_text += abst.text
                abstract_text += '\n'
        authors = []
        affiliations = set()
        for author in article.iter('Author'):
            if author:
                author_forename = author.find('ForeName')
                forename = author_forename.text if author_forename is not None else None
                author_lastname = author.find('LastName')
                lastname = author_lastname.text if author_lastname is not None else None

                author_name = None
                if lastname and not forename:
                    author_name = lastname
                elif forename and not lastname:
                    author_name = forename
                elif forename and lastname:
                    author_name = forename + ' ' + lastname

                if author_name:
                    authors.append(author_name)
                
                affiliation_info = author.find('AffiliationInfo')
                if affiliation_info:
                    for aff in affiliation_info.findall('Affiliation'):
                        affiliations.add(aff.text)
    
        mesh_headings = []
        for mesh_heading in article.iter('MeshHeading'):
            descriptor_text = mesh_heading.find('DescriptorName').text
            qualifers = mesh_heading.findall('QualifierName')
            qualifers_text = None
            if qualifers is not None:
                qualifers_text = [q.text for q in qualifers]
            # qualifers_text = [q.text for q in qualifers] if qualifers is not None else None
            mesh_heading_dict = {'descriptor': descriptor_text, 'qualifiers': qualifers_text}
            mesh_headings.append(mesh_heading_dict)
                   


        source = article.find('MedlineCitation').find('Article').find('Journal')
        source_title = source.find('Title').text
    
        source_info = source.find('JournalIssue')
        medium = source_info.get('CitedMedium')
        source_volume = source_info.find('Volume')
        volume = source_volume.text if source_volume is not None else None
        source_issue = source_info.find('Issue')
        issue = source_issue.text if source_issue is not None else None
        
        source_pubdate = source_info.find('PubDate')
        pubyear = None
        pubmonth = None
        if source_pubdate:
            source_pubyear = source_pubdate.find('Year')
            pubyear = int(source_pubyear.text) if source_pubyear is not None else None
            
            source_pubmonth = source_pubdate.find('Month')
            pubmonth = source_pubmonth.text if source_pubmonth is not None else None
    
        source_pagination = list(article.iter('MedlinePgn'))
        pagination = None
        if source_pagination:
            pagination = source_pagination[0].text if source_pagination[0].text != 'UNKNOWN' else None
    
        source_ids = list(article.iter('ArticleId'))
        doi = None
        for sid in source_ids:
            if sid.get('IdType') == 'doi':
                doi = sid.text
                break
    
        medline_journal_info = list(article.iter('MedlineJournalInfo'))
        country = None
        source_title_abbrev = None
        if medline_journal_info:
            source_country = medline_journal_info[0].find('Country')
            if source_country is not None:
                if source_country.text.upper() != 'UNKNOWN':
                    country = source_country.text.upper() 
            medlineTA = medline_journal_info[0].find('MedlineTA')
            source_title_abbrev = medlineTA.text if medlineTA is not None else None
    
        publication_types = []
        for pub_type in article.iter('PublicationType'):
            publication_types.append(pub_type.text)
        # article_json = {}
        # article_json['title'] = title
        # article_json['authors'] = authors
        # article_json['abstract'] = abstract
        # article_json['meshHeadings'] = list(meshHeadings)
        # source_json = {}
        # source_json['title'] = source_title
        # source_json['issue'] = source_issue
        # source_json['volume'] = source_volume
        # source_json['pagination'] = source_pagination
        # source_json['pubDate'] = {'year': source_pubyear, 'month': source_pubmonth}
        article_json = {
            'title': title,
            'authors': authors,
            'abstract': abstract_text,
            'affiliations': list(affiliations),
            'mesh_headings': mesh_headings,
            'source': {
                'types': publication_types,
                'medium': medium,
                'title': source_title,
                'title_abbrev': source_title_abbrev,
                'volume': volume,
                'issue': issue,
                'pagination': pagination,
                'pub_date': {
                    'year': pubyear,
                    'month': pubmonth
                },
                'doi': doi,
                'country': country
            }
        }
        # print(json.dumps(article_json, sort_keys=False, indent=4))
    
        pmid = article.find('MedlineCitation').find('PMID').text
        total_json[pmid] = article_json
    outname = os.path.splitext(os.path.basename(filename))[0] + '.json'
    outpath = os.path.join(todir, outname)
    with open(outpath, 'w') as outfile:
        json.dump(total_json, outfile)
    


print('end parsing...')
