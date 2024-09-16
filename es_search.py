from elasticsearch import Elasticsearch
import pandas as pd
from luqum.parser import parser
from luqum.elasticsearch import ElasticsearchQueryBuilder
from luqum.elasticsearch import SchemaAnalyzer

es = Elasticsearch()


def retrieve(q, entity='keywords'):
    error = None
    df = None
    boolean_operators = ['AND', 'OR', 'NOT']
    boolean_search = False
    for boolean in boolean_operators:
        if q.find(boolean) > -1:
            boolean_search = True
    dsl = {}
    if boolean_search:
        dsl = boolean_handler(q)
    elif entity == 'genes':
        if q.startswith('"') and q.endswith('"'):
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
                                    'boost': 2
                                }
                            }
                            },
                            {'match_phrase': {
                                'abstract': {
                                    'query': q,
                                    'boost': 3
                                }
                            }
                            },
                            {'match_phrase': {
                                'authors': {
                                    'query': q,
                                    'boost': 1
                                }
                            }
                            },
                        ],
                        'must': [
                            {'match_phrase': {'mesh_headings.qualifers': 'genetics'}}
                        ]
                    }

                }
            }
        else:
            dsl = {
                'query': {
                    'bool': {
                        'must': [
                            # {'match': {'meshHeadings': 'genetics'}},
                            {
                                'nested': {
                                    'path': 'mesh_headings',
                                    'query': {
                                        'bool': {
                                            'must': [
                                                {"match": {
                                                    "mesh_headings.qualifers": "genetics"}}
                                            ]
                                        }
                                    }
                                }
                            },

                            # {'match': {'title': q}},
                            {'multi_match': {
                                'query': q,
                                'fields': ['title^3', 'abstract^2', 'authors^1'],
                                # if term has multiple words return documents that have all words in title or abstract
                                'operator': 'AND'
                            }}
                        ]
                        # 'should': [
                        #     # {'match': {
                        #     #     'title': {
                        #     #         'query': q,
                        #     #         'boost': 2
                        #     #     }
                        #     # }},
                        #     {'match': {
                        #         'abstract': {
                        #             'query': q,
                        #             'boost': 3
                        #         }
                        #     }},
                        #     {'match': {
                        #         'authors': {
                        #             'query': q,
                        #             'boost': 1
                        #         }
                        #     }
                        #     }
                        # ]
                    }
                }
            }
    else:
        # if query is in quotations match only if terms are in the same order, consecutively
        if q.startswith('"') and q.endswith('"'):
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
                                'abstract': {
                                    'query': q,
                                    'boost': 2
                                }
                            }
                            },
                            {'match_phrase': {
                                'authors': {
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
                        'fields': ['title^3', 'abstract^2', 'authors^1'],
                        # if term has multiple words return documents that have all words in title or abstract
                        'operator': 'AND'
                    }}
            }
    resp = es.search(index='pubmed2', doc_type="_doc", body=dsl, size=300)
    if q == '':
        error = 'Please provide search terms'
    else:
        pubmed_dict = {}
        df = pd.DataFrame()
        for res in resp['hits']['hits']:
            authors = []
            affiliations = []
            for a in res['_source']['authors']:
                authors.append(a)
            if None in authors:
                authors.remove(None)
            for a in res['_source']['affiliations']:
                affiliations.append(a)
            affiliations = [i for i in affiliations if i is not None]
            title = res['_source']['title']
            abstract = res['_source']['abstract']
            pmid = res['_id']
            url = 'https://www.ncbi.nlm.nih.gov/pubmed/?term=' + pmid
            pubdate = res['_source']['source']['pubDate']['month']
            pubYear = res['_source']['source']['pubDate']['year']
            journal = res['_source']['source']['title']
            journal_abrev = res['_source']['source']['title_abbrev']
            volume = res['_source']['source']['volume']
            issue = res['_source']['source']['issue']
            doi = res['_source']['source']['doi']
            pages = res['_source']['source']['pagination']
            meshHeadings = res['_source']['mesh_headings']
            data = {'authors': authors,
                    'affiliations': affiliations,
                    'title': title,
                    'abstract': abstract,
                    'pmid': pmid,
                    'url': url,
                    'pubdate': pubdate,
                    'journal': journal,
                    'journal_abbrev': journal_abrev,
                    'volume': volume,
                    'issue': issue,
                    'doi': doi,
                    'pages': pages,
                    'meshHeadings': meshHeadings,
                    'pubYear': pubYear}
            series = pd.Series(data)
            df = df.append(series, ignore_index=True)
    return df, error


def boolean_handler(q):
    array = lswb(q)
    string_with_fields = add_fields(array)
    query_tree = parse(string_with_fields)
    query_tree = add_all_fields(query_tree)
    es_dsl = {}
    es_dsl['query'] = query_tree
    return es_dsl


def lswb(q):
    q_array = []
    index = 0
    open_parentheses = False
    for i in range(len(q)):
        if i == len(q) - 1 and q[i] != ')':
            q_array.append(q[index:i+1])
        elif q[i] == '(':
            open_parentheses = True
            index = i
        elif open_parentheses:
            if q[i] == ')':
                q_array.append(q[index:i+1])
                open_parentheses = False
                index = i + 1
        elif i != len(q) - 1:
            if q[i+1] == ' ' and (q[i+2:i+5] == 'AND' or q[i+2:i+5] == 'NOT' or q[i+2:i+4] == 'OR'):
                q_array.append(q[index:i+1])
                index = i + 1
            elif q[i-3:i] == 'AND' or q[i-3:i] == 'NOT':
                q_array.append(q[i-3:i])
                index = i + 1
            elif q[i-2:i] == 'OR':
                q_array.append(q[i-2:i])
                index = i + 1
    return q_array


def parse(q):
    schema = {
        "pubmed2": {
            "mappings": {
                "properties": {
                    "abstract": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "affiliations": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "authors": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "meshHeadings": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "num_cited_by": {
                        "type": "long"
                    },
                    "source": {
                        "properties": {
                            "country": {
                                "type": "text",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            },
                            "doi": {
                                "type": "text",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            },
                            "issue": {
                                "type": "text",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            },
                            "medium": {
                                "type": "text",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            },
                            "pagination": {
                                "type": "text",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            },
                            "pubDate": {
                                "properties": {
                                    "month": {
                                        "type": "text",
                                        "fields": {
                                            "keyword": {
                                                "type": "keyword",
                                                "ignore_above": 256
                                            }
                                        }
                                    },
                                    "year": {
                                        "type": "text",
                                        "fields": {
                                            "keyword": {
                                                "type": "keyword",
                                                "ignore_above": 256
                                            }
                                        }
                                    }
                                }
                            },
                            "title": {
                                "type": "text",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            },
                            "title_abbrev": {
                                "type": "text",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            },
                            "types": {
                                "type": "text",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            },
                            "volume": {
                                "type": "text",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            }
                        }
                    },
                    "title": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    }
                }
            }
        }
    }
    schema_analizer = SchemaAnalyzer(schema)
    message_es_builder = ElasticsearchQueryBuilder(
        **schema_analizer.query_builder_options())
    tree = parser.parse(q)
    query = message_es_builder(tree)
    return query


def add_fields(arr):
    string = ''
    for word in arr:
        if word.startswith('(') and word.endswith(')'):
            text = word[1:-1]
            new_array = lswb(text)
            new_string = add_fields(new_array)
            string += "(" + new_string + ")"
        elif word not in ['NOT', 'AND', 'OR']:
            string += 'title:' + word
        else:
            string += " " + word + " "
    return string


def add_all_fields(dsl):
    dsl_copy = dsl
    dsl_keyword = next(iter(dsl_copy['bool']))
    for index, ele in enumerate(dsl_copy['bool'][dsl_keyword]):
        if next(iter(ele)) == 'match':
            query = ele['match']['title']['query']
            dsl_copy['bool'][dsl_keyword][index] = {'multi_match': {
                'query': query,
                'fields': ['title^3', 'abstract^2', 'authors^1'],
                # if term has multiple words return documents that have all words in title or abstract
                'operator': 'AND'
            }}
        elif next(iter(ele)) == 'match_phrase':
            query = ele['match_phrase']['title']['query']
            dsl_copy['bool'][dsl_keyword][index] = {
                'bool': {
                    'should': [
                        {'match_phrase': {'title': {'query': query}}},
                        {'match_phrase': {
                            'abstract': {'query': query}}},
                        {'match_phrase': {'author': {'query': query}}}
                    ]
                }
            }
        elif next(iter(ele)) == 'bool':
            dsl_copy['bool'][dsl_keyword][index] = add_all_fields(ele)
    return dsl_copy
