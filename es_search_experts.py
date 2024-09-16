from elasticsearch import Elasticsearch
import pandas as pd

es = Elasticsearch()


def retrieve(q):
    error = None
    df = None
    if q == '': # if query is empty, get all documents
        dsl = {
            "query": {
                "match_all": {}
            }
        }
    # if query is in quotations match only if terms are in the same order, consecutively
    elif q.startswith('"') and q.endswith('"'):
        # remove quotation marks from query
        q = q.strip('\"')
        dsl = {
            "query": {
                # bool applies mutiple queries
                'bool': {
                    # return match if phrase is in title OR in abstract
                    'should': [
                        {'match_phrase': {
                            'expert_title': {
                                'query': q,
                                'boost': 2
                            }
                        }
                        },
                        {'match_phrase': {
                            'expert_title_description': {
                                'query': q,
                                'boost': 2
                            }
                        }
                        },
                        {'match_phrase': {
                            'areas_expertise': {
                                'query': q,
                                'boost': 2
                            }
                        }
                        },
                        {'match_phrase': {
                            'industry_expertise': {
                                'query': q,
                                'boost': 2
                            }
                        }
                        },
                        {'match_phrase': {
                            'biography': {
                                'query': q,
                                'boost': 1
                            }
                        }
                        },
                    ]
                }

            }
        }
    # if query is not in quotations, match if document has term in title OR abstract OR authors
    else:
        dsl = {
            "query": {
                'multi_match': {
                    'query': q,
                    'fields': ['expert_title^2', 'expert_title_description^2', 'areas_expertise^2', 'industry_expertise^2','biography^1'],
                    # if term has multiple words return documents that have all words in title or abstract
                    'operator': 'AND'
                }}
        }
    resp = es.search(index='experts', doc_type="_doc", body=dsl, size=300)
    
    num_docs = resp['hits']['total']['value']

    pubmed_dict = {}
    df = pd.DataFrame()
    for res in resp['hits']['hits']:
        affiliations = []
        industry_expertise = []
        areas_expertise = []
        education = []

        name = res['_source']['name']
        expert_title = res['_source']['expert_title']
        expert_title_description = res['_source']['expert_title_description']
        biography = res['_source']['biography']
        url = res['_source']['url']

        for a in res['_source']['affiliations']:
            affiliations.append(a)
        affiliations = [i for i in affiliations if i is not None]

        for i in res['_source']['industry_expertise']:
            industry_expertise.append(i)
        industry_expertise = [i for i in industry_expertise if i is not None]

        for a in res['_source']['areas_expertise']:
            areas_expertise.append(a)
        areas_expertise = [i for i in areas_expertise if i is not None]

        for e in res['_source']['education']:
            education.append(a)
        education = [i for i in education if i is not None]
        
        data = {'name': name,
                'expert_title': expert_title,
                'expert_title_description': expert_title_description,
                'biography': biography,
                'industry_expertise': industry_expertise,
                'areas_expertise': areas_expertise,
                'affiliations': affiliations,
                'education': education,
                'url': url}
        series = pd.Series(data)
        df = df.append(series, ignore_index=True)
    return df, num_docs, error
