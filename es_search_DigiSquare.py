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
                            'title': {
                                'query': q,
                                'boost': 3
                            }
                        }
                        },
                        {'match_phrase': {
                            'content': {
                                'query': q,
                                'boost': 3
                            }
                        }
                        },
                        {'match_phrase': {
                            'subject': {
                                'query': q,
                                'boost': 2
                            }
                        }
                        },
                        {'match_phrase': {
                            'type': {
                                'query': q,
                                'boost': 2
                            }
                        }
                        },
                        {'match_phrase': {
                            'audience': {
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
                    'fields': ['title^3', 'content^3', 'subject^2', 'type^2','audience^1'],
                    # if term has multiple words return documents that have all words in title or abstract
                    'operator': 'AND'
                }}
        }
    resp = es.search(index='digitalsquare', doc_type="_doc", body=dsl, size=300)

    num_docs = resp['hits']['total']['value']

    pubmed_dict = {}
    df = pd.DataFrame()
    for res in resp['hits']['hits']:

        title = res['_source']['title']
        content = res['_source']['content']
        sponsorship = res['_source']['sponsorship']
        url = res['_source']['url']
        contributor = res['_source']['contributor']
        abstract = res['_source']['abstract']
        provenance = res['_source']['provenance']
        subject = res['_source']['subject']
        type = res['_source']['type']
        coverage = res['_source']['coverage']
        accessioned_date = res['_source']['accessioned_date']
        available_date = res['_source']['available_date']
        audience = res['_source']['audience']
        file_name = res['_source']['file_name']




        data = {'title': title,
                'content': content,
                'sponsorship': sponsorship,
                'url': url,
                'contributor': contributor,
                'abstract': abstract,
                'subject': subject,
                'provenance': provenance,
                'type': type,
                'coverage':coverage,
                'accessioned_date': accessioned_date,
                'available_date':available_date,
                'audience': audience,
                'file_name':file_name}

        series = pd.Series(data)
        df = df.append(series, ignore_index=True)
    return df, num_docs, error
