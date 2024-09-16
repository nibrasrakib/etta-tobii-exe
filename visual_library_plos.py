# coding: utf-8 

"""
Visual library
"""

import logging
import argparse
import sys
import os
import re
import operator
import math
import gzip
import bz2
import copy
import random
import csv
import html
import chardet

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

import pandas as pd
# from beakerx import *
import itertools

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.manifold import _t_sne

from my_tsne import MyTSNE

from IPython.display import display, HTML

# regular expression patterns
tokenize = re.compile("[^\w\-]+")  # a token is composed of
# alphanumerics or hyphens
exclude = re.compile("(\d+)$")    # numbers

# nltk lemmatizer
'''
import nltk
from multiprocessing import Pool
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
'''

'''
helpers
'''


def is_gene_name(gene_set, keyword):
    # return True
    # return re.match('^[a-zA-Z]+[0-9]+', keyword)
    return keyword in gene_set


def load_gene_data():
    filename = os.path.join((os.path.dirname(os.path.abspath(
        __file__))), 'data/genes_non_hypothetical2.csv')
    df = pd.read_csv(filename)
    # print(df.head())

    symbol_list = df['Symbol'].values.tolist()
    symbol_results = [str(w) for w in symbol_list]
    # print(symbol_results[0:10])

    alias_raw_list = df['Aliases'].values.tolist()

    alias_nested_list = [str(words).split(', ') for words in alias_raw_list]
    alias_results = list(itertools.chain.from_iterable(alias_nested_list))
    # print(alias_results[0:10])

    total_results = set(symbol_results + alias_results)
    # for i, val in enumerate(itertools.islice(total_results, 10)):
    #     print(val)
    return total_results


'''
main
'''


def main():

    data_dir = 'data'
    csv_dir = 'csv'

    # log output
    logging.basicConfig(filename='.visual_library.log',
                        format='%(asctime)s : %(levelname)s'
                        ': %(message)s', level=logging.INFO)

    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rank", type=int, default=5,
                        help="Consider the top R ranked tokens in each "
                        "document (default: 5)")
    parser.add_argument("-d", "--p_docs", type=float, default=0.5,
                        help="The percentage of documents that must "
                        "contain a token ranked above R for the token "
                        "to be selected (default: 0.5)")
    parser.add_argument("--mindf", type=int, default=1,
                        help="Ignore terms below mindf "
                        "(default: 1)")
    parser.add_argument("--theta", type=float, default=0.9,
                        help="Theta for maxi-min clustering. "
                        "The greater the theta, the larger the clusters."
                        "(default: 0.9)")
    parser.add_argument("-i", "--input", default='data/inspec-corrected.tsv',
                        help="Input file (default: data/inspec-corrected.tsv)")
    parser.add_argument("-w", "--weight", default='tfidf',
                        help="Term weighting. Either 'tfidf' or "
                        "'binary' (default: tfidf)")
    parser.add_argument("-m", "--matrix", default=None,
                        help="Output file name for produced "
                        "term-document matrix (default: None)")
    parser.add_argument("-s", "--sim", default=None,
                        help="Output file name for similarity "
                        "matrix (default: None)")
    parser.add_argument("-c", "--cluster", default="document",
                        help="Which to cluster, document or term "
                        "(default: document)")
    parser.add_argument("--clustering", default="maximin",
                        help="Clustering algorithm, maximin or kmeans "
                        "(default: maximin)")
    parser.add_argument("-k", "--n_clusters", type=int, default=10,
                        help="Number of clusters (k) for k-means. "
                        "Not used for maximin "
                        "(default: 10)")
    parser.add_argument("--svd", type=int, default=0,
                        help="Number of components in applying SVD. "
                        "Not applied if 0 "
                        "(default: 0)")
    parser.add_argument("-f", "--fields",
                        default="",
                        help="Text fields to be used. One or any "
                        "combination of title, abstract, and body. "
                        "If not specified, all fields are used "
                        "(default: \"\")")
    parser.add_argument("--mesh", default=None,
                        help="Mesh term file corresponding to "
                        "the input. Needed for PMC file "
                        "(default: None)")
    parser.add_argument('--single',
                        help='Evaluate only single-class instances.'
                        '(default: False)',
                        action='store_true')
    parser.add_argument('--balance',
                        help='Balance the data. (default: False)',
                        action='store_true')

    args = parser.parse_args()
    logging.info(args)

    # Read stopword list
    stopwords = read_stopwords()

    # Balance the data
    if args.balance and re.search("(plos|pmc|med)", args.input):
        balance_data(file=args.input)
        input_file = ".balanced_"+args.input
    else:
        input_file = args.input

    # Read documents
    print("Reading documents...")
    docs, df, w2id, mesh = read_documents(data_dir,
                                          input=input_file,
                                          stopwords=stopwords,
                                          fields=args.fields,
                                          single_class=args.single)
    print("Finished reading %d documents" % len(docs))
    print("%d terms were identified" % len(df))

    # Read MeSH file if provided
    if args.mesh:
        print("Reading MeSH file...")
        mesh = read_mesh(args.mesh)

    # Remove terms whose df is lower than mindf
    inf = []
    if args.mindf:
        for w in df:
            if df[w] <= args.mindf:
                inf.append(w)
        for w in inf:
            del df[w]
    print("%d terms were removed" % len(inf))

    # Compute tfidf and find key terms
    print("Computing TFIDF and finding key terms...")
    docs, dfr = compute_tfidf(docs, df, args.weight, args.rank)

    # Output matrix if output name is specified
    if args.matrix:
        print("Writing out TFIDF matrix...")
        output_matrix(csv_dir, args.matrix, docs, df.keys())

    # Sort and output results (discovered keywords)
    keywords = output_keywords(len(docs), dfr, df, args.p_docs)

    # Create new matrix with the keywords (mesh is also needed
    # in case some docs are removed)
    docs, mesh = update(docs, keywords, mesh)

    # clustering
    print()
    print("Clustering...")
    if args.clustering == "maximin":
        _, membership, _, sc, sct = \
            maximin(csv_dir, docs, args.sim,
                    args.cluster, keywords, np.array(mesh).ravel(),
                    args.theta, args.svd)
        # visualize_network(sim, keywords, membership)
    elif args.clustering == "kmeans":
        # print(args.cluster,docs.shape[0]) #here
        membership, _, _, _, sc, sct = kmeans(docs, args.cluster, keywords,
                                              args.svd, args.n_clusters,
                                              np.array(mesh).ravel())
        # visualize_network(sim, keywords, membership)

    if args.cluster == "document" and len(mesh) > 0:
        print(" Silhouette   = %f" % sc)
        print(" Silhouette_t = %f" % sct)
        evaluate(mesh, membership)


'''
make balanced data
'''
# escape special characters
# regular expression
regex = re.compile('\s+')

# unescape special characters


def unescape(str):
    str = str.replace('&amp;', '&')
    str = str.replace('&apos;', "'")
    str = str.replace('&quot;', '"')
    str = str.replace('&gt;', '>')
    str = str.replace('&lt;', '<')
    return str


def escape(str):
    str = str.replace('&', ' &amp; ')
    str = str.replace('\'', ' &apos; ')
    str = str.replace('"', ' &quot; ')
    str = str.replace('>', ' &gt; ')
    str = str.replace('<', ' &lt; ')
    return str


def balance_data(file=None):

    # store data temporarily
    data = dict()

    # read file
    with open_by_suffix(file) as f:
        for line in f:
            if "plos" in file or "pmc" in file:
                _, _, _, _, m = \
                    line.rstrip().split('\t')
            else:
                _, _, _, m, _ = line.split('\t')

            # for skipping multi-class instances
            m = m.split('|')
            if len(m) > 1:
                continue

            if m[0] in data:
                data[m[0]].append(line)
            else:
                data[m[0]] = [line]

    min = sys.maxsize
    for k in data:
        if len(data[k]) < min:
            min = len(data[k])

    # write
    with gzip.open(".balanced_"+file, "wt") as f:
        for k in data:
            f.write(''.join(data[k][0:min]))


'''
Compute purity (macro-average is Javed's version of homogeneity)
'''


def compute_purity(mesh, membership):

    # add ids to true labels (mesh). only the first
    # cluster label is considered.
    m2id = {m: i for i, m in enumerate(set([x[0] for x in mesh]))}
    id2m = {i: m for m, i in m2id.items()}
    labels = [m2id[m[0]] for m in mesh]

    # confusion matrix
    cm = metrics.confusion_matrix(labels, membership)

    # compute max
    sum_max = sum(cm.max(axis=0))

    # find largest match
    k2c = {i: x for i, x in enumerate(cm.argmax(axis=0))}
    preds = [k2c[x] for x in membership]
    nclus = len(set(preds))

    # compute
    sum_h = 0.0
    for i in range(nclus):
        h = cm[k2c[i], i]/cm[:, i].sum()
        sum_h += h
        # print(i, "%.3f" % (h))

    return sum_h/nclus, sum_max/len(membership)


'''
Evaluation
'''


def evaluate(mesh, membership):

    if len(mesh) == 0:
        print("No labels (MeSH) provided. "
              "Cannot evaluate the clusters.")
        return

    # precision (Javed's version of homogeneity)
    prt_macro, prt_micro = compute_purity(mesh, membership)
    print(" Purity-macro = %f" % prt_macro)
    print(" Purity-micro = %f" % prt_micro)

    # v-score (variant for multilabels)
    c = compute_completeness(mesh, membership)
    h = compute_homogeneity(mesh, membership)
    vd = (2*h*c)/(h+c)
    print(" VD-score     = %f" % vd)

    # other metrics

    # treat a multi-labeled instance as multiple instances
    preds = []
    labels = []
    for i, l in enumerate(mesh):
        for l_ in l:
            labels.append(l_)
            preds.append(membership[i])

    # compute
    v = metrics.v_measure_score(labels, preds)
    ai = metrics.adjusted_rand_score(labels, preds)
    ami = metrics.adjusted_mutual_info_score(labels, preds)
    fms = metrics.fowlkes_mallows_score(labels, preds)

    print(" V-score      = %f" % v)
    print(" A-RAND-I     = %f" % ai)
    print(" A-MI         = %f" % ami)
    print(" FMS          = %f" % fms)

    return c, h, vd, v, ai, ami, fms, prt_macro, prt_micro


'''
File opener depending on suffix
'''


def open_by_suffix(filename):
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt', encoding='UTF-8')
    elif filename.endswith('.bz2'):
        return bz2.BZ2file(filename, 'r')
    else:  # assume text file
        return open(filename, 'r')


'''
Read MeSH files
'''


def read_mesh(file_name):
    mesh = []
    with open(file_name) as f:
        for line in f:
            meshes = line.rstrip().split('|')
            tmp = []
            for m in meshes:
                tmp.append(m.split('/')[0])
            mesh.append(tmp)

    return mesh


'''
Compute completeness
'''


def compute_completeness(labels, pred):

    C = set()
    K = set(pred)

    # count
    N = float(0)
    a = dict()
    for i, l in enumerate(labels):
        for l_ in l:
            N += 1 / len(l)
            C.add(l_)
            key = l_+str(pred[i])
            if key in a:
                a[key] += 1 / len(l)
            else:
                a[key] = 1 / len(l)

    # compute H(K)
    hk = 0
    for k in K:
        s = 0
        for c in C:
            # print(c, k)
            key = c+str(k)
            if key in a:
                # print(a[key])
                s += a[key]
        s /= N
        hk += s * math.log(s, 2)
    hk = -hk
    # print("hk = %f" % hk)

    # compute H(K|C)
    hkc = 0
    for c in C:
        nc = 0
        for k in K:
            key = c+str(k)
            if key in a:
                nc += a[key]
        for k in K:
            # print(c, k)
            key = c+str(k)
            if key in a:
                # print(a[key])
                hkc += a[key]/N * math.log(a[key]/nc, 2)
    hkc = -hkc
    # print("hkc = %f" % hkc)
    c = 1 - hkc/hk
    print(" Completeness = %f" % c)

    return c


'''
Compute homogeneity
'''


def compute_homogeneity(labels, pred):

    C = set()
    K = set(pred)

    # count
    N = float(0)
    a = dict()
    for i, l in enumerate(labels):
        for l_ in l:
            N += 1 / len(l)
            C.add(l_)
            key = l_+str(pred[i])
            if key in a:
                a[key] += 1 / len(l)
            else:
                a[key] = 1 / len(l)

    # compute H(C)
    hc = 0
    for c in C:
        nc = 0
        for k in K:
            # print(c, k)
            key = c+str(k)
            if key in a:
                # print(a[key])
                nc += a[key]
        nc /= N
        hc += nc * math.log(nc, 2)
    hc = -hc
    # print("hc = %f" % hc)

    # compute H(C|K)
    hck = 0
    for k in K:
        ac = 0
        for c in C:
            key = c+str(k)
            if key in a:
                ac += a[key]
        for c in C:
            # print(c, k)
            key = c+str(k)
            if key in a:
                # print(a[key])
                hck += a[key]/N * math.log(a[key]/ac, 2)
    hck = -hck
    # print("hck = %f" % hck)
    h = 1 - hck/hc
    print(" Homogeneity  = %f" % h)

    return h


def convert_sparse(doc_term_mat, keywords):

    # add ids to keywords
    keywords.sort()
    w2id = {c: i for i, c in enumerate(keywords)}

    # Convert to scipy matrix for faster calculation
    data = []
    row_idx = []
    col_idx = []
    for i in range(len(doc_term_mat)):
        data += doc_term_mat[i].values()
        col_idx += [w2id[w] for w in doc_term_mat[i].keys()]
        row_idx += [i] * len(doc_term_mat[i])

    data = csr_matrix((data, (row_idx, col_idx)),
                      (len(doc_term_mat), len(keywords)))

    return data


'''
kmeans
'''


def kmeans(doc_term_mat, keywords, org_ids, n_components, k, n_desc=5):
    print("kmeans")
    # print(k)
    # print(doc_term_mat.shape)
    # print(doc_term_mat)
    # add ids to keywords
    # keywords.sort()
    # w2id = {c:i for i,c in enumerate(keywords)}

    # print('pre transpose')
    error_r = None
    # in case k is smaller than num of doc_term_mat
    k = min(k, doc_term_mat.shape[0])
    # print('pre normalize')
    doc_term_mat = doc_term_mat / norm(doc_term_mat, axis=1)[:, np.newaxis]
    term_doc_mat = np.transpose(doc_term_mat)
    # print('post normalize')
    # print(term_doc_mat)

    # column dimensionality of the transposed data
    col_dim = term_doc_mat.shape[1]
    # to use TruncatedSVD later, n_components must be strictly less than the # of features, in this case, the dimentionality of columns (documents)
    n_components = min(n_components, col_dim - 1)

    # SVD
    if n_components == 0:  # no svd
        km = KMeans(init='k-means++', n_clusters=k, n_init=10,
                    random_state=0)
        km.fit(term_doc_mat)

        # set cluster centers to get descriptions
        centers = km.cluster_centers_

    else:
        try:
            term_doc_mat = np.asarray(term_doc_mat)
            
            
            # Print size of term_doc_mat
            print("Size of term_doc_mat: ")
            print(term_doc_mat.shape)
            # Print a snippet of term_doc_mat for inspection
            print(f"Snippet of term_doc_mat: {term_doc_mat[:5, :5]}")
            # Print non-zero values in term_doc_mat
            print(f"Non-zero values in term_doc_mat: {np.count_nonzero(term_doc_mat)}")
            if np.isnan(term_doc_mat).any() or np.isinf(term_doc_mat).any():
                print("Input matrix contains NaN or infinite values.")
                raise ValueError("Input matrix contains NaN or infinite values.")
            term_term_mat = cosine_similarity(term_doc_mat)
            print("Size of term_term_mat: "
                  + str(term_term_mat.shape))
        except:
            error_r = "Problems with cosine similarity matrix!"
            return None, None, None, None, error_r

        try:
            km = KMeans(init='k-means++', n_clusters=k,
                        n_init=10, random_state=0).fit(term_term_mat)
        except:
            error_r = "Problems kmeans fitting!"
            return None, None, None, None, error_r
        print("Kmeans fitting done.")
    cluster_labels = []
    cluster_label_ids = []
    # get cluster descriptions
    for j, c in enumerate(km.cluster_centers_):
        # in case n_desc is greater than num of keywords
        n_desc_ = min(n_desc, len(c)-1)
        i = np.argpartition(-c, n_desc_)[:n_desc_]  # top n (unsorted)
        c_i = c[i]
        i_ = np.argsort(c_i)[::-1]  # sort
        k_i = np.array(keywords)[i]
        labels_ = k_i[i_].tolist()
        labels_ = [i for i in labels_ if len(i) > 3]
        # print("C%d: " % (j+1) + ", ".join(labels_))
        cluster_labels.append(labels_)
        cluster_label_ids.append(i)
        # print("Cluster {}: {}".format(j, ' '.join(cluster_labels)))
        # print("what I want to see:")
        # print(cluster_labels)

    total_num_docs = doc_term_mat.shape[0]
    # (num of doc * k) matrix, k represents num of clusters; measuring the score each doc gets in each cluster, in the form of an array of vectors
    doc_cluster_coords = [[0] * k for i in range(total_num_docs)]

    # # print(cluster_label_ids)
    # for i, lbls in enumerate(cluster_label_ids):  # i is cluster id
    #     for lbl in lbls:
    #         docs_for_lbl = term_doc_mat[lbl].tolist()[0]

    #         for j, tfidf in enumerate(docs_for_lbl):  # j is doc index
    #             doc_cluster_coords[j][i] += tfidf
    try:
        for i, lbls in enumerate(cluster_label_ids):  # i is cluster id
            for lbl in lbls:
                # Ensure that term_doc_mat[lbl] returns an array
                docs_for_lbl = term_doc_mat[lbl]

                if docs_for_lbl.ndim == 1:
                    docs_for_lbl = docs_for_lbl.tolist()
                else:
                    docs_for_lbl = docs_for_lbl.flatten().tolist()

                for j, tfidf in enumerate(docs_for_lbl):  # j is doc index
                    doc_cluster_coords[j][i] += tfidf

    except TypeError as e:
        print(f"TypeError encountered: {e}")
        error_r = "Problems with document cluster coordinates!"

    membership = [coord.index(max(coord)) for coord in doc_cluster_coords]

    from collections import defaultdict
    # count cluster members
    id2members = defaultdict(list)
    for i, m in enumerate(membership):
        bib_id = org_ids[i]
        id2members[m].append(bib_id)

    id2members = sharing_algorithm(
        k, org_ids, id2members, doc_cluster_coords, total_num_docs)
    # sort id2members in ascending order by key so that it starts from 0; then convert to defaultdict to enable default initialization
    id2members = defaultdict(list, dict(
        sorted(id2members.items(), key=operator.itemgetter(0))))

    # get cluster ids with members and only keep cluster labels to for these clusters
    populated_ids = []
    cluster_labels_ = []
    for id in id2members:
        if id2members[id]:
            populated_ids.append(id)
            cluster_labels_.append(cluster_labels[id])

    # re-map cluster ids so that intermittent 'populated_ids' become contiguous ids
    id2members_ = {}
    index = 0
    for k, v in id2members.items():
        id2members_[index] = v
        index += 1

    # print('id2members_: ')
    # print(id2members_)

    # if n_components > 0:
    #     print("Formed %d clusters after reducing "
    #           "to %d dimensions." % (k, n_components))
    # else:
    #     print("Formed %d clusters w/o SVD." % k)

    # get 2d coordinates by t-SNE
    # Number of samples
    n_samples = len(km.cluster_centers_[populated_ids])

    # Set perplexity to a value less than the number of samples
    perplexity = min(30, n_samples - 1)  # Example: set perplexity to 30 or less if n_samples is less than 30

    tsne = MyTSNE(n_components=2,
                  perplexity=perplexity,
                  max_iter=5000, random_state=123)
    cluster_centers = tsne.fit_transform(
        km.cluster_centers_[populated_ids])  # only calculate tsne for the populated_ids

    '''
    # get 2d coordinates by svd
    svd_model = TruncatedSVD(n_components=2,
                             random_state=42)
    cluster_centers = svd_model.fit_transform(km.cluster_centers_)
    '''

    # scale to (0,1)
    scaler = preprocessing.MinMaxScaler()
    cluster_centers = scaler.fit_transform(cluster_centers).tolist()

    # scale intermediate data too
    coordinates = scaler.transform(tsne.coordinates.reshape(-1, 2)).\
        reshape(tsne.coordinates.shape).tolist()
    print("Returning from kmeans")
    return id2members_, cluster_centers,\
        cluster_labels_, coordinates, error_r


def sharing_algorithm(k, org_ids, id2members, doc_cluster_coords, total_num_docs):

    id2members_ = id2members.copy()
    # how many documents each cluster has
    raw_id2freq = {k: len(v) for k, v in id2members_.items()}

    unpopulated_cluster_ids = []
    for i in range(k):  # for all the cluster ids
        if i not in raw_id2freq:  # does not belong to the clusters who own documents
            unpopulated_cluster_ids.append(i)
            raw_id2freq[i] = 0  # give this cluster a spot

    if unpopulated_cluster_ids:  # if there are clusters who have zero documents assigned, steal docs from "the wealthy" to "the poor"
        q = compute_q(k)  # select the top q scores from each doc coordinate
        landscape = compute_landscape(doc_cluster_coords, q)
        assignment_distribution = get_assign_num(
            unpopulated_cluster_ids, total_num_docs, q, landscape)

        doc_index_list = list(range(total_num_docs))
        for cluster_id, num_assigned in assignment_distribution.items():
            # score_diff_dict is {doc_index : the coord number diff between current cluster id and the highest one ...}
            score_diff_dict = {}
            for i in doc_index_list:
                coord = landscape[i]
                if cluster_id in coord:
                    score_diff_dict[i] = list(coord.values())[
                        0] - coord[cluster_id]
            score_diff_dict = dict(sorted(score_diff_dict.items(), key=operator.itemgetter(
                1), reverse=True))  # sort score_diff_dict in descending order by value
            num_shared = 0
            for doc_index_to_share in score_diff_dict:
                if num_shared < num_assigned:  # only share the amount assigned
                    id2members_[cluster_id].append(org_ids[doc_index_to_share])
                    num_shared += 1

    return id2members_

def compute_q(num_cls):
    if num_cls >= 8:
        return 3
    elif num_cls >= 5:
        return 2
    else:
        return 1

def get_membership_and_cluster_labels(km, keywords, doc_term_mat):

    membership = km.labels_.tolist()

    from collections import defaultdict
    id2members = defaultdict(list)
    for i, m in enumerate(membership):
        id2members[m].append(i)

    # sort by key in ascending value
    id2members = dict(sorted(id2members.items(), key=operator.itemgetter(0)))
    cluster_labels = []

    for cluster_id, doc_indices in id2members.items():
        cluster_term_coord = (doc_term_mat.shape[1]) * [0]
        for i in doc_indices:
            tfidf_list = doc_term_mat[i].toarray().tolist()[0]
            for j, tfidf in enumerate(tfidf_list):
                cluster_term_coord[j] += tfidf

        num_non_zero_term = 0
        for coord in cluster_term_coord:
            if coord > 0:
                num_non_zero_term += 1
        n_desc_ = min(n_desc, num_non_zero_term - 1)
        coord = np.array(cluster_term_coord)
        # get the indices (cluster ids) of the largest scores
        top_term_indices = np.argpartition(coord, -n_desc_)[-n_desc_:]
        sorted_indices = top_term_indices[np.argsort(
            coord[top_term_indices])[::-1]]
        top_terms = [keywords[ind] for ind in sorted_indices]
        cluster_labels.append(top_terms)
        # print("C%d: " % (cluster_id+1) + ", ".join(top_terms))

    return membership, cluster_labels


def compute_landscape(doc_cluster_coords, q):
    # landscape is the portion of doc_cluster_coords in which each coord only contains the top three coord numbers in the original full size coord
    landscape = []
    for doc_coord in doc_cluster_coords:
        coord = np.array(doc_coord)
        # get the indices (cluster ids) of the q largest scores
        indices = np.argpartition(coord, -q)[-q:]
        sorted_indices = indices[np.argsort(coord[indices])[::-1]]
        top_ind2score = {i: coord[i] for i in sorted_indices}
        landscape.append(top_ind2score)

    return landscape


def get_assign_num(cluster_ids, total_num_docs, q, landscape):
    from collections import defaultdict
    # score_distribution is {cluster id : count in landscape ...}
    score_distribution = defaultdict(int)
    for coord_dict in landscape:  # coord_dict is {cluster id : score ...}
        for k, v in coord_dict.items():
            score_distribution[k] += 1

    total_count = total_num_docs * q
    # proportions is {cluster id : percentage ...}
    proportions = {i: score_distribution[i] / total_count for i in cluster_ids}

    # assignment_distribution is {cluster id : num of docs it will have...}
    assignment_distribution = defaultdict(int)

    for i in cluster_ids:
        # select ceiling interger as the num of doc to assign to current cluster so that at least 1 doc is assigned
        num_docs_assigned = math.ceil(proportions[i] * total_num_docs)
        assignment_distribution[i] = num_docs_assigned

    return assignment_distribution


def distribute(total_num_docs, doc_cluster_coords):
    from collections import defaultdict
    q = 3  # select the top q score from each doc coordinate
    landscape = []
    for doc_coord in doc_cluster_coords:
        coord = np.array(doc_coord)
        # get the indices (cluster ids) of the 3 largest scores
        indices = np.argpartition(coord, -q)[-q:]
        sorted_indices = indices[np.argsort(coord[indices])[::-1]]
        top_ind2score = {i: coord[i] for i in sorted_indices}
        landscape.append(top_ind2score)

    # score_distribution is {cluster id : count in landscape ...}
    score_distribution = defaultdict(int)
    for coord_dict in landscape:  # coord_dict is {cluster id : score ...}
        for k, v in coord_dict.items():
            score_distribution[k] += 1

    # sort score_distribution is ascending order by value
    score_distribution = dict(
        sorted(score_distribution.items(), key=operator.itemgetter(1)))
    total_count = total_num_docs * q
    # proportions is {cluster id : percentage ...}
    proportions = {i: score_distribution[i] /
                   total_count for i in score_distribution}
    # assignment_distribution is {cluster id : num of docs it will have...}
    assignment_distribution = defaultdict(int)

    num_docs_remain = total_num_docs
    for i in range(len(proportions) - 1):  # for all the cluster ids except the last one
        # select floor interger as the num of doc to assign to current cluster
        num_docs_assigned = math.floor(proportions[i] * total_num_docs)
        assignment_distribution[i] = num_docs_assigned
        num_docs_remain -= num_docs_assigned

    # the biggest cluster gets the rest of docs
    assignment_distribution[-1] = num_docs_remain

    doc_index_list = list(range(total_num_docs))

    id2members = defaultdict(list)
    for cluster_id, num_of_docs in assignment_distribution.items():
        score_diff_dict = defaultdict(float)
        for i in doc_index_list:
            if i == -1:  # this doc index has been taken
                continue
            coord = landscape[i]
            if cluster_id in coord:
                score_diff_dict[i] = list(coord.values())[
                    0] - coord[cluster_id]
        score_diff_dict = dict(sorted(score_diff_dict.items(), key=operator.itemgetter(
            1), reverse=True))  # sort score_diff_dict is descending order by value
        kvpairs = list(score_diff_dict.items())
        for kv in kvpairs[:num_of_docs]:
            doc_index_list[kv[0]] = -1  # invalidate this doc index
            id2members[cluster_id].append(kv[0])

    membership_dict = {}
    for cluster_id, doc_indices in id2members.items():
        for index in doc_indices:
            membership_dict[index] = cluster_id

    membership = list(membership_dict.values())
    return membership


'''
Read stopwords
'''


def read_stopwords(data_dir="data", filename="stopwords.txt"):
    print("Reading stopwords...")
    stopwords = set()
    with open(os.path.join(data_dir, filename)) as f:
        for term in f:
            if term.startswith("#"):
                continue
            stopwords.add(term.rstrip())
    return stopwords

def generate_dynamic_stopwords(body):
    terms = tokenize.split(body)
    from collections import Counter
    counter = Counter(terms)

    top_terms = counter.most_common(int(len(counter) * 0.01)) # the top 1% most common terms
    stopwords = set()
    for k, v in top_terms:
        stopwords.add(k)

    return stopwords

'''
Read documents from simplejson instance (for SOLR)
'''


def read_json(input, dataset, stopwords=[]):
    print("Reading read_json...")
    docs = []  # store documents
    df = dict()  # document frequency
    w2id = dict()
    cnt_w = 0
    bibs = []

    for d in input['response']['docs']:
        title_ = d['title']
        if len(d['author']) < 3:
            author_ = ', '.join(d['author'])
        else:
            author_ = ', '.join(d['author'][0:2])  # first two
            author_ += ', ..., '+d['author'][-1]  # last
        bibs.append({'title': title_,
                     'journal': d['journal_name'],
                     'author': author_,
                     'pdate': d['publication_date'][:10]})

        terms = tokenize.split(
            (title_+" "+" ".join(d['abstract'])).lower())
        '''
        terms = [lemmatizer.lemmatize(w) for w in terms if w != '' and \
                     w not in stopwords and \
                     not exclude.match(w)]
        '''
        terms = [w for w in terms if w != '' and
                 w not in stopwords and
                 not exclude.match(w)]
        # count df
        for w in set(terms):
            if w in df:
                df[w] += 1
            else:
                df[w] = 1
                w2id[w] = cnt_w
                cnt_w += 1
        # count tf
        tf = dict()
        for w in terms:
            if w in tf:
                tf[w] += 1
            else:
                tf[w] = 1
        docs.append(tf)

    return docs, df, w2id, bibs


'''
Read documents from pysolr results
'''


def read_pysolr(results, dataset, stopwords=[]):
    print("Reading pysolr...")
    docs = []  # store documents
    df = dict()  # document frequency
    w2id = dict()
    cnt_w = 0
    bibs = []

    if dataset == "NYTIMES":  # can be cleaned
        for d in results:
            text = ""
            title_ = d['title']
            body_ = d['body']
            bibs.append({'title': title_,
                         'body': body_,
                         'html': d['html']})
            text += title_
            text += body_
            if 'body' in d:
                text += " " + " ".join(d["body"])  # newly added
            if text == "":
                continue
            terms = tokenize.split(text.lower())
            '''
            terms = [lemmatizer.lemmatize(w) for w in terms if w != '' and \
                         w not in stopwords and \
                         not exclude.match(w)]
            '''
            terms = [w for w in terms if w != '' and
                     w not in stopwords and
                     not exclude.match(w)]
            # print("lenth of terms")
            # print(len(terms))
            # count df
            for w in set(terms):
                if w in df:
                    df[w] += 1
                else:
                    df[w] = 1
                    w2id[w] = cnt_w
                    cnt_w += 1
            # count tf
            tf = dict()
            for w in terms:
                if w in tf:
                    tf[w] += 1
                else:
                    tf[w] = 1
            docs.append(tf)

        return docs, df, w2id, bibs

    else:
        for d in results:
            if not 'publication_date' in d:
                text = ""
                pmid_ = 'PMID not Provided'
                title_ = 'Title not Provided'
                author_ = 'Author Name not Provided'
                if 'title' in d:
                    title_ = d['title']
                if 'author' in d:
                    if len(d['author']) < 3:
                        author_ = ', '.join(d['author'])
                    else:
                        author_ = ', '.join(d['author'][0:2])  # first two
                        author_ += ', ..., '+d['author'][-1]  # last
                if 'pmid' in d:
                    pmid_ = d['pmid']
                bibs.append({'title': title_,
                             'author': author_,
                             'pmid': pmid_})
                text += title_
                if 'abstract' in d:
                    text += " " + " ".join(d["abstract"])

                if "body" in d:
                    text += " " + d["body"]

                if text == "":
                    continue
                terms = tokenize.split(text.lower())
                '''
                terms = [lemmatizer.lemmatize(w) for w in terms if w != '' and \
                             w not in stopwords and \
                             not exclude.match(w)]
                '''
                terms = [w for w in terms if w != '' and
                         w not in stopwords and
                         not exclude.match(w)]
                # count df
                for w in set(terms):
                    if w in df:
                        df[w] += 1
                    else:
                        df[w] = 1
                        w2id[w] = cnt_w
                        cnt_w += 1
                # count tf
                tf = dict()
                for w in terms:
                    if w in tf:
                        tf[w] += 1
                    else:
                        tf[w] = 1
                docs.append(tf)

            else:
                text = ""
                pmid_ = 'PMID not Provided'
                title_ = 'Title not Provided'
                author_ = 'Author Name not Provided'
                journal_name_ = 'Journal Name not Provided'
                if 'title' in d:
                    title_ = d['title']
                if 'author' in d:
                    if len(d['author']) < 3:
                        author_ = ', '.join(d['author'])
                    else:
                        author_ = ', '.join(d['author'][0:2])  # first two
                        author_ += ', ..., '+d['author'][-1]  # last
                if 'journal_name' in d:
                    journal_name_ = d['journal_name']
                if 'pmid' in d:
                    pmid_ = d['pmid']
                bibs.append({'title': title_,
                             'journal': journal_name_,
                             'author': author_,
                             'pmid': pmid_,
                             'pdate': d['publication_date'][:10]})
                text += title_
                if 'abstract' in d:
                    text += " " + " ".join(d["abstract"])

                if "body" in d:
                    text += " " + d["body"]

                if text == "":
                    continue
                terms = tokenize.split(text.lower())
                '''
                terms = [lemmatizer.lemmatize(w) for w in terms if w != '' and \
                             w not in stopwords and \
                             not exclude.match(w)]
                '''
                terms = [w for w in terms if w != '' and
                         w not in stopwords and
                         not exclude.match(w)]
                # count df
                for w in set(terms):
                    if w in df:
                        df[w] += 1
                    else:
                        df[w] = 1
                        w2id[w] = cnt_w
                        cnt_w += 1
                # count tf
                tf = dict()
                for w in terms:
                    if w in tf:
                        tf[w] += 1
                    else:
                        tf[w] = 1
                docs.append(tf)

        return docs, df, w2id, bibs


'''
Read documents
'''  # need to change


def read_documents(data_dir, input=None, source=None,
                   stopwords=[], fields="", single_class=False):
    print("Reading documents...")
    docs = []  # store documents
    df = dict()  # document frequency
    w2id = dict()
    cnt_w = 0

    # get file name
    if source:
        file = os.path.join(data_dir, source)
    else:
        file = input
    print("file to check")
    print(file)
    mesh = []
    with open_by_suffix(file) as f:

        for line in f:
            if "inspec" in file:
                terms = tokenize.split(line.lower())
            elif "plos" in file or "pmc" in file:
                _, title, abs, body, m = \
                    line.rstrip().split('\t')
                text = ""
                if "title" in fields:
                    text += title
                if "abstract" in fields:
                    text += " " + abs
                if "body" in fields:
                    text += " " + body
                if text == "":
                    text = title+" "+abs+" "+body
                terms = tokenize.split(text.lower())
                m = m.split('|')

                # for skipping multi-class instances
                if single_class and len(m) > 1:
                    continue

                mesh.append(m)
            else:
                pmid, title, abs, m, _ = line.split('\t')
                terms = tokenize.split((title+" "+abs).lower())
                m = m.split('|')

                # for skipping multi-class instances
                if single_class and len(m) > 1:
                    continue

                tmp = []
                for m_ in m:
                    tmp.append(m_.split('/')[0])
                mesh.append(tmp)

            terms = [w for w in terms if w not in stopwords and
                     len(w) > 1 and not exclude.match(w)]
            # count df
            for w in set(terms):
                if w in df:
                    df[w] += 1
                else:
                    df[w] = 1
                    w2id[w] = cnt_w
                    cnt_w += 1

            # count tf
            tf = dict()
            for w in terms:
                if w in tf:
                    tf[w] += 1
                else:
                    tf[w] = 1

            docs.append(tf)

    return docs, df, w2id, mesh


'''
Read uploaded plain text
'''


def read_plaintext(path, uploaded_file, q, dataset, stopwords=[]):
    print("Reading uploaded plain text...")
    path_to_file = path + '/' + uploaded_file
    docs = []  # store documents
    result_file_list = []
    df = dict()  # document frequency
    w2id = dict()
    cnt_w = 0
    q_and = []
    q_or = []
    q_str = ''
    if '"' in q and 'AND' not in q:
        q_str = q[1:-1]
        q_str = q.lower()
    elif 'AND' in q:
        q = q.split(" AND ")
        for term in q:
            if '"' in term:
                q_and.append(term[1:-1].lower())
            else:
                q_and.append(term.lower())
    elif 'OR' in q:
        q = q.split(" OR ")
        for term in q:
            if '"' in term:
                q_or.append(term[1:-1].lower())
            else:
                q_or.append(term.lower())
    else:
        q = tokenize.split(q.lower())  # needs to be specified
    print(q)

    bibs = []

    if '.txt' in uploaded_file:
        text = open(path_to_file, "r", encoding="utf-8")
        # print(path)
        del_menu = np.load(os.path.join(path, 'del_menu.npy'),
                           allow_pickle=True).item()
        # print("del_menu")
        # print(del_menu)
        delimiter = del_menu[uploaded_file]
        if delimiter == "no delimiter provided":
            delimiter = '\n\n'
        # delimiter = text.readlines()[-1]

        # text = open(path_to_file,"r",encoding="utf-8")

        print("delimiter = " + delimiter)
        # print(text)
        # text = open(path,"r")
        # print(path)
        text = text.read()
        # print(type(text))

        # print("text"+text)
        file_list = text.split(delimiter)
        # print(len(file_list))
        # result_file_text = ''
        # print("len(file_list)")
        # print(len(file_list))
    elif '.csv' in uploaded_file:
        file_list = []
        with open(path_to_file, newline='', encoding="utf-8") as csvfile:
            filereader = csv.reader(csvfile)
            for row in filereader:
                file_list.append((''.join(row)))

    for file in file_list:
        # print(file)
        if q_and != []:
            score = 0
            for term in q_and:
                if term in tokenize.split(file.lower()):
                    score += 1
            if score == len(q_and):
                result_file_list.append(file)
        elif q_or != []:
            for term in q_or:
                if term in tokenize.split(file.lower()):
                    result_file_list.append(file)
        elif q_str != '':
            if q_str in tokenize.split(file.lower()):
                result_file_list.append(file)
        else:
            for term in q:
                # print(term)
                if term in tokenize.split(file.lower()):
                    # result_file_text += file
                    # print(len(result_file_text))
                    # print(len(text))
                    result_file_list.append(file)
    # result_file_text = ' '.join(result_file_list)
    # print("len(result_file_list)")
    # print(len(result_file_list))
    for file in result_file_list:
        body_toshow = html.unescape(file)
        # body_toshow = body_toshow.replace('\'',' ')
        # body_toshow = body_toshow.replace('\n', ' _startinganewline_ ')
        # print("file is")
        # print(re.findall('"',body_toshow))
        bibs.append({'title': file[:50]+'...',
                     'body_toshow': body_toshow})
        terms = tokenize.split(file.lower())
        '''
        terms = [lemmatizer.lemmatize(w) for w in terms if w != '' and \
                     w not in stopwords and \
                     not exclude.match(w)]
        '''
        terms = [w for w in terms if w != '' and
                 w not in stopwords and
                 not exclude.match(w)]  # and \
        # w not in q]
        # print(len(terms))
        # count df
        for w in set(terms):
            if w in df:
                df[w] += 1
            else:
                df[w] = 1
                w2id[w] = cnt_w
                cnt_w += 1
        # count tf
        tf = dict()
        for w in terms:
            if w in tf:
                tf[w] += 1
            else:
                tf[w] = 1
        docs.append(tf)
    return docs, df, w2id, bibs


def read_df(dataframe, dataset, stopwords=[]):
    print("Reading dataframe...")
    doc_term_mat = []  # store documents
    df = dict()  # document frequency
    w2id = dict()
    cnt_w = 0
    bibs = []
    for index, row in dataframe.iterrows():
        string = ''
        bibs_dict = {}
        if 'pmid' in dataframe:
            pmid_ = str(row['pmid'])
            bibs_dict['pmid'] = pmid_
        if 'title' in dataframe:
            # print('title in dataframe')
            title_ = str(row['title'])
            bibs_dict['title'] = title_
            string = string + title_.lower() + ' '
        if 'author' in dataframe:
            # print('author in dataframe')
            author_ = str(row['author'])
            bibs_dict['author'] = author_
        if 'authors' in dataframe:
            # print('author in dataframe')
            author_ = row['authors']
            bibs_dict['author'] = author_
        if 'affiliations' in dataframe:
            # print('affiliation dataframe')
            affiliation_ = row['affiliations']
            bibs_dict['affiliations'] = affiliation_
        if 'source' in dataframe:
            source_ = str(row['source'])
            bibs_dict['journal'] = source_
        if 'abstract' in dataframe:
            # print('abstract in dataframe')
            abstract_ = str(row['abstract'])
            # TAG: for gene coverage testing
            bibs_dict['abstract'] = abstract_
            string = string + abstract_.lower() + ' '
        if 'description' in dataframe:
            # print("description in dataframe")
            description_ = str(row['description'])
        if 'snippet' in dataframe:
            # print("description in dataframe")
            description_ = str(row['snippet'])
            # bibs_dict['description']=description_
            string = string + description_.lower() + ' '
        if 'content' in dataframe:
            content_ = str(row['content'])
            # bibs_dict['content']=content_
            string = string + content_.lower() + ' '
        if 'url' in dataframe:
            url_ = str(row['url'])
            bibs_dict['html'] = url_
        if 'pubdate' in dataframe:
            pubdate_ = str(row['pubdate'])
            bibs_dict['pubdate'] = pubdate_
        if 'pubYear' in dataframe:
            pubYear_ = str(row['pubYear'])
            bibs_dict['pubYear'] = pubYear_
        if 'journal' in dataframe:
            journal_ = str(row['journal'])
            bibs_dict['journal'] = journal_
        if 'journal_abbrev' in dataframe:
            journalabbrev_ = str(row['journal_abbrev'])
            bibs_dict['journal_abbrev'] = journalabbrev_
        if 'volume' in dataframe:
            volume_ = str(row['volume'])
            bibs_dict['volume'] = volume_
        if 'issue' in dataframe:
            issue_ = str(row['issue'])
            bibs_dict['issue'] = issue_
        if 'doi' in dataframe:
            doi_ = str(row['doi'])
            bibs_dict['doi'] = doi_
        if 'pages' in dataframe:
            pages_ = str(row['pages'])
            bibs_dict['pages'] = pages_
        if 'meshHeadings' in dataframe:
            meshHeadings_ = row['meshHeadings']
            bibs_dict['meshHeadings'] = meshHeadings_

        bibs.append(bibs_dict)
        # print("bibs_dict")
        # print(bibs_dict)
        # print("string is")
        # print(string)
        terms = tokenize.split(string)

        # terms = [lemmatizer.lemmatize(w) for w in terms if w != '' and \
        #             w not in stopwords and \
        #             not exclude.match(w)]

        terms = [w for w in terms if w != '' and
                 w not in stopwords and
                 not exclude.match(w)]
        # count df
        for w in set(terms):
            if w in df:
                df[w] += 1
            else:
                df[w] = 1
                w2id[w] = cnt_w
                cnt_w += 1
        # count tf
        tf = dict()
        for w in terms:
            if w in tf:
                tf[w] += 1
            else:
                tf[w] = 1
        doc_term_mat.append(tf)

    return doc_term_mat, df, w2id, bibs

def read_df_experts(dataframe, dataset, stopwords=[]):
    print("Reading dataframe_experts...")
    doc_term_mat = []  # store documents
    df = dict()  # document frequency
    w2id = dict()
    cnt_w = 0
    bibs = []

    for index, row in dataframe.iterrows():
        string = ''
        bibs_dict = {}
        if 'name' in dataframe:
            name_ = str(row['name'])
            bibs_dict['title'] = name_

        if 'expert_title' in dataframe:
            expert_title_ = str(row['expert_title'])
            string = string + expert_title_.lower() + ' '
            bibs_dict['expert_title'] = expert_title_

        if 'expert_title_description' in dataframe:
            expert_title_description_ = str(row['expert_title_description'])
            string = string + expert_title_description_.lower() + ' '
            bibs_dict['expert_title_description'] = expert_title_description_

        if 'affiliations' in dataframe:
            affiliation_ = row['affiliations']
            bibs_dict['affiliations'] = affiliation_

        if 'biography' in dataframe:
            biography_ = str(row['biography'])
            string = string + biography_.lower() + ' '
            bibs_dict['abstract'] = biography_

        if 'industry_expertise' in dataframe:
            industry_expertise_ = row['industry_expertise']
            string = string + ' '.join(str(x).lower() for x in industry_expertise_) + ' '
            bibs_dict['industry_expertise'] = industry_expertise_

        if 'areas_expertise' in dataframe:
            areas_expertise_ = row['areas_expertise']
            string = string + ' '.join(str(x).lower() for x in areas_expertise_) + ' '
            bibs_dict['areas_expertise'] = areas_expertise_

        if 'url' in dataframe:
            url_ = str(row['url'])
            bibs_dict['html'] = url_


        bibs.append(bibs_dict)
        terms = tokenize.split(string)

        terms = [w for w in terms if w != '' and
                 w not in stopwords and
                 not exclude.match(w)]
        # count df
        for w in set(terms):
            if w in df:
                df[w] += 1
            else:
                df[w] = 1
                w2id[w] = cnt_w
                cnt_w += 1
        # count tf
        tf = dict()
        for w in terms:
            if w in tf:
                tf[w] += 1
            else:
                tf[w] = 1
        doc_term_mat.append(tf)

    return doc_term_mat, df, w2id, bibs

def read_df_digi(dataframe, dataset, stopwords=[]):
    print("Reading dataframe_digi...")
    doc_term_mat = []  # store documents
    df = dict()  # document frequency
    w2id = dict()
    cnt_w = 0
    bibs = []

    for index, row in dataframe.iterrows():
        string = ''
        bibs_dict = {}
        if 'title' in dataframe:
            title_ = str(row['title'])
            bibs_dict['title'] = title_

        if 'content' in dataframe:
            content_ = str(row['content'])
            string = string + content_.lower() + ' '
            #bibs_dict['content'] = content_

        if 'sponsorship' in dataframe:
            sponsorship_ = str(row['sponsorship'])
            bibs_dict['sponsorship'] = sponsorship_

        if 'coverage' in dataframe:
            coverage_ = str(row['coverage'])
            bibs_dict['coverage'] = coverage_

        if 'subject' in dataframe:
            subject_ = str(row['subject'])
            string = string + subject_.lower() + ' '
            bibs_dict['subject'] = subject_

        if 'type' in dataframe:
            type_ = str(row['type'])
            string = string + type_.lower() + ' '
            bibs_dict['type'] = type_

        if 'audience' in dataframe:
            audience_ = str(row['audience'])
            string = string + audience_.lower()+ ' '
            bibs_dict['audience'] = audience_

        if 'url' in dataframe:
            url_ = str(row['url'])
            bibs_dict['html'] = url_

        if 'contributor' in dataframe:
            contributor_ = str(row['contributor'])
            bibs_dict['contributor'] = contributor_

        if 'accessioned_date' in dataframe:
            accessioned_date_ = str(row['accessioned_date'])
            bibs_dict['accessioned_date'] = accessioned_date_

        if 'available_date' in dataframe:
            available_date_ = str(row['available_date'])
            bibs_dict['available_date'] = str(row['available_date'])

        if 'file_name' in dataframe:
            file_name_ = str(row['file_name'])
            bibs_dict['file_name'] = file_name_

        if 'provenance' in dataframe:
            provenance_ = str(row['provenance'])
            bibs_dict['provenance'] = provenance_

        if 'abstract' in dataframe:
            abstract_ = str(row['abstract'])
            bibs_dict['abstract'] = abstract_

        bibs.append(bibs_dict)
        terms = tokenize.split(string)

        terms = [w for w in terms if w != '' and
                 w not in stopwords and
                 not exclude.match(w)]
        # count df
        for w in set(terms):
            if w in df:
                df[w] += 1
            else:
                df[w] = 1
                w2id[w] = cnt_w
                cnt_w += 1
        # count tf
        tf = dict()
        for w in terms:
            if w in tf:
                tf[w] += 1
            else:
                tf[w] = 1
        doc_term_mat.append(tf)

    return doc_term_mat, df, w2id, bibs


def read_df_genes(gene_set, dataframe, dataset, stopwords=[]):
    print("Reading dataframe_genes...")
    doc_term_mat = []  # store documents
    df = dict()  # document frequency
    w2id = dict()
    cnt_w = 0
    bibs = []

    for index, row in dataframe.iterrows():
        string = ''
        bibs_dict = {}
        if 'title' in dataframe:
            title_ = str(row['title']) if row['title'] else 'N/A'
            bibs_dict['title'] = title_
            string = string + title_ + ' '
        if 'author' in dataframe:
            # print('author in dataframe')
            author_ = str(row['author'])
            bibs_dict['author'] = author_
        if 'authors' in dataframe:
            # print('author in dataframe')
            author_ = row['authors']
            bibs_dict['author'] = author_
        if 'affiliations' in dataframe:
            affiliation_ = str(row['affiliations']
                               ) if row['affiliations'] else 'N/A'
            bibs_dict['affiliations'] = affiliation_
        if 'source' in dataframe:
            source_ = str(row['source']) if row['source'] else 'N/A'
            bibs_dict['journal'] = source_
        if 'abstract' in dataframe:
            abstract_ = str(row['abstract']) if row['abstract'] else 'N/A'
            # TAG: for gene coverage testing
            bibs_dict['abstract'] = abstract_
            string = string + abstract_ + ' '
        description_ = None
        if 'description' in dataframe:
            description_ = str(row['description'])
        if 'snippet' in dataframe:
            description_ = str(row['snippet'])
            # bibs_dict['description']=description_
        if description_:
            string = string + description_ + ' '
        if 'content' in dataframe:
            content_ = str(row['content'])
            # bibs_dict['content']=content_
            string = string + content_ + ' '
        if 'url' in dataframe:
            url_ = str(row['url']) if row['url'] else 'N/A'
            bibs_dict['html'] = url_
        if 'pubdate' in dataframe:
            pubdate_ = str(row['pubdate'])
            bibs_dict['pubdate'] = pubdate_
        if 'pubYear' in dataframe:
            pubYear_ = str(row['pubYear'])
            bibs_dict['pubYear'] = pubYear_
        if 'journal' in dataframe:
            journal_ = str(row['journal'])
            bibs_dict['journal'] = journal_
        if 'journal_abbrev' in dataframe:
            journalabbrev_ = str(row['journal_abbrev'])
            bibs_dict['journal_abbrev'] = journalabbrev_
        if 'volume' in dataframe:
            volume_ = str(row['volume'])
            bibs_dict['volume'] = volume_
        if 'issue' in dataframe:
            issue_ = str(row['issue'])
            bibs_dict['issue'] = issue_
        if 'doi' in dataframe:
            doi_ = str(row['doi'])
            bibs_dict['doi'] = doi_
        if 'pages' in dataframe:
            pages_ = str(row['pages'])
            bibs_dict['pages'] = pages_
        if 'meshHeadings' in dataframe:
            meshHeadings_ = row['meshHeadings']
            bibs_dict['meshHeadings'] = meshHeadings_
        # print("bibs_dict")
        # print(bibs_dict)
        # print("string is")
        # print(string)
        terms = tokenize.split(string)

        terms = [w for w in terms if w != '' and w != 'N/A' and
                 is_gene_name(gene_set, w) and
                 w not in stopwords and
                 not exclude.match(w)]
        # to add bibs with no gene terms or not to add, that's a question...
        if not terms:
            continue
        bibs.append(bibs_dict)
        # print("TERMS are")
        # print(terms)
        # print('\n')
        # count df
        for w in set(terms):
            if w in df:
                df[w] += 1
            else:
                df[w] = 1
                w2id[w] = cnt_w
                cnt_w += 1
        # count tf
        tf = dict()
        for w in terms:
            if w in tf:
                tf[w] += 1
            else:
                tf[w] = 1
        doc_term_mat.append(tf)

    return doc_term_mat, df, w2id, bibs


def read_df_authors(dataframe, dataset, stopwords=[]):
    print("Reading dataframe_authors...")
    doc_term_mat = []  # store documents
    df = dict()  # document frequency
    w2id = dict()
    cnt_w = 0
    bibs = []

    for index, row in dataframe.iterrows():
        terms = []
        bibs_dict = {}
        if 'title' in dataframe:
            title_ = str(row['title'])
            bibs_dict['title'] = title_
        if 'author' in dataframe:
            author_ = str(row['author'])
            bibs_dict['author'] = author_
        if 'authors' in dataframe:
            author_ = row['authors']
            bibs_dict['author'] = [a for a in author_ if a != None]
            terms = [a for a in author_ if a != None]
        if 'affiliations' in dataframe:
            affiliation_ = row['affiliations']
            bibs_dict['affiliations'] = affiliation_
        if 'source' in dataframe:
            source_ = str(row['source'])
            bibs_dict['journal'] = source_
        if 'abstract' in dataframe:
            abstract_ = str(row['abstract'])
            # TAG: for gene coverage testing
            bibs_dict['abstract'] = abstract_
        if 'description' in dataframe:
            description_ = str(row['description'])
        if 'snippet' in dataframe:
            description_ = str(row['snippet'])
            # bibs_dict['description']=description_
        if 'content' in dataframe:
            content_ = str(row['content'])
            # bibs_dict['content']=content_
        if 'url' in dataframe:
            url_ = str(row['url'])
            bibs_dict['html'] = url_
        if 'meshHeadings' in dataframe:
            meshHeadings_ = row['meshHeadings']
            bibs_dict['meshHeadings'] = meshHeadings_
        if 'pubdate' in dataframe:
            pubdate_ = str(row['pubdate'])
            bibs_dict['pubdate'] = pubdate_
        if 'pubYear' in dataframe:
            pubYear_ = str(row['pubYear'])
            bibs_dict['pubYear'] = pubYear_
        if 'journal' in dataframe:
            journal_ = str(row['journal'])
            bibs_dict['journal'] = journal_
        if 'journal_abbrev' in dataframe:
            journalabbrev_ = str(row['journal_abbrev'])
            bibs_dict['journal_abbrev'] = journalabbrev_
        if 'volume' in dataframe:
            volume_ = str(row['volume'])
            bibs_dict['volume'] = volume_
        if 'issue' in dataframe:
            issue_ = str(row['issue'])
            bibs_dict['issue'] = issue_
        if 'doi' in dataframe:
            doi_ = str(row['doi'])
            bibs_dict['doi'] = doi_
        if 'pages' in dataframe:
            pages_ = str(row['pages'])
            bibs_dict['pages'] = pages_

        bibs.append(bibs_dict)
        # print("bibs_dict")
        # print(bibs_dict)
        # print("string is")
        # print(string)
        # print(terms)

        # count df
        for w in set(terms):
            if w in df:
                df[w] += 1
            else:
                df[w] = 1
                w2id[w] = cnt_w
                cnt_w += 1
        # count tf
        tf = dict()
        # num_authors = len(terms)
        # if num_authors == 0:
        #     continue
        # step = 1/num_authors
        # # first and last author get most tf, authors in middle decrease by order
        # for i in range(num_authors - 1):
        #     author = terms[i]
        #     tf[author] = 1 - step * i

        # last_author = terms[-1]
        # tf[last_author] = 1

        for w in terms:
            tf[w] = 1
        # give more weight to the first author
        if len(terms) > 0:
            tf[terms[0]] = 3
        doc_term_mat.append(tf)

    return doc_term_mat, df, w2id, bibs


def process_experts_by_topic(dataframe, dataset, num_cls, stopwords=[]):
    print("Processing experts by topic...")
    doc_term_mat, df, w2id, bibs = read_df_experts(dataframe, dataset, stopwords)


    # Remove terms whose df is lower than mindf
    MINDF = 1
    if MINDF > 0:
        inf = []
        for w in df:
            if df[w] <= MINDF:
                inf.append(w)
        for w in inf:
            del df[w]

    # save original matrix and df
    orig_doc_term_mat = doc_term_mat
    orig_df = df

    doc_term_mat, dfr = compute_tfidf(doc_term_mat, df, rank=5)

    keywords = output_keywords(len(doc_term_mat), dfr, df, p_docs=1.0)

    doc_term_mat, org_ids = update(doc_term_mat, keywords)

    doc_term_mat = convert_sparse(doc_term_mat, keywords)
    id2members, cluster_centers, cluster_desc, coordinates, error = \
        kmeans(doc_term_mat, keywords, org_ids, n_components=20, k=num_cls, n_desc=15)


    return orig_doc_term_mat, orig_df, df, dfr, keywords, cluster_centers, cluster_desc, coordinates, id2members, bibs, org_ids

def process_experts_by_name(dataframe, dataset, num_cls, stopwords=[]):
    print("Processing experts by name...")
    doc_term_mat, df, w2id, bibs = read_df_experts(dataframe, dataset, stopwords)


    # Remove terms whose df is lower than mindf
    MINDF = 1
    if MINDF > 0:
        inf = []
        for w in df:
            if df[w] <= MINDF:
                inf.append(w)
        for w in inf:
            del df[w]

    # save original matrix and df
    orig_doc_term_mat = doc_term_mat
    orig_df = df

    doc_term_mat, dfr = compute_tfidf(doc_term_mat, df, rank=5)

    keywords = output_keywords(len(doc_term_mat), dfr, df, p_docs=1.0)

    doc_term_mat, org_ids = update(doc_term_mat, keywords)

    doc_term_mat = convert_sparse(doc_term_mat, keywords)
    id2members, cluster_centers, coordinates, error = \
        kmeans_doc_doc(doc_term_mat, keywords, org_ids, n_components=20, k=num_cls, n_desc=15)

    cluster_desc = []
    for i, m in id2members.items():
        desc = [bibs[j]['title'] for j in m[:15]] # get the first 15 expert names as cluster labels
        cluster_desc.append(desc)

    return orig_doc_term_mat, orig_df, df, dfr, keywords, cluster_centers, cluster_desc, coordinates, id2members, bibs, org_ids

def process_digi_by_topic(dataframe, dataset, num_cls, stopwords=[]):
    print("Processing digi by topic...")
    doc_term_mat, df, w2id, bibs = read_df_digi(dataframe, dataset, stopwords)


    # Remove terms whose df is lower than mindf
    MINDF = 1
    if MINDF > 0:
        inf = []
        for w in df:
            if df[w] <= MINDF:
                inf.append(w)
        for w in inf:
            del df[w]

    # save original matrix and df
    orig_doc_term_mat = doc_term_mat
    orig_df = df

    doc_term_mat, dfr = compute_tfidf(doc_term_mat, df, rank=5)

    keywords = output_keywords(len(doc_term_mat), dfr, df, p_docs=1.0)

    doc_term_mat, org_ids = update(doc_term_mat, keywords)

    doc_term_mat = convert_sparse(doc_term_mat, keywords)
    id2members, cluster_centers, cluster_desc, coordinates, error = \
        kmeans(doc_term_mat, keywords, org_ids, n_components=20, k=num_cls, n_desc=15)


    return doc_term_mat, orig_df, df, dfr, keywords, cluster_centers, cluster_desc, coordinates, id2members, bibs, org_ids

def read_df_authors_2(dataframe, dataset, stopwords=[]):
    print("Reading dataframe_authors...")

    doc_term_mat = []
    # author_names = set()
    df = dict()  # document frequency
    w2id = dict()
    cnt_w = 0
    bibs = []
    terms_total = []
    from collections import defaultdict
    author_doc_dict = defaultdict(list)
    author_term_dict = defaultdict(lambda: defaultdict(int))

    i = 0
    for index, row in dataframe.iterrows():
        string = ''
        bibs_dict = {}
        if 'title' in dataframe:
            title_ = str(row['title'])
            bibs_dict['title'] = title_
            string = string + title_.lower() + ' '
        if 'authors' in dataframe:
            author_ = row['authors']
            bibs_dict['author'] = [a for a in author_ if a != None]
        if 'affiliations' in dataframe:
            affiliation_ = row['affiliations']
            bibs_dict['affiliations'] = affiliation_
        if 'source' in dataframe:
            source_ = str(row['source'])
            bibs_dict['journal'] = source_
        if 'abstract' in dataframe:
            abstract_ = str(row['abstract'])
            # TAG: for gene coverage testing
            bibs_dict['abstract'] = abstract_
            string = string + abstract_.lower() + ' '
        if 'description' in dataframe:
            description_ = str(row['description'])
        if 'snippet' in dataframe:
            description_ = str(row['snippet'])
            # bibs_dict['description']=description_
            string = string + description_.lower() + ' '
        if 'content' in dataframe:
            content_ = str(row['content'])
            # bibs_dict['content']=content_
            string = string + content_.lower() + ' '
        if 'url' in dataframe:
            url_ = str(row['url'])
            bibs_dict['html'] = url_
        if 'meshHeadings' in dataframe:
            meshHeadings_ = row['meshHeadings']
            bibs_dict['meshHeadings'] = meshHeadings_
        if 'pubdate' in dataframe:
            pubdate_ = str(row['pubdate'])
            bibs_dict['pubdate'] = pubdate_
        if 'pubYear' in dataframe:
            pubYear_ = str(row['pubYear'])
            bibs_dict['pubYear'] = pubYear_
        if 'journal' in dataframe:
            journal_ = str(row['journal'])
            bibs_dict['journal'] = journal_
        if 'journal_abbrev' in dataframe:
            journalabbrev_ = str(row['journal_abbrev'])
            bibs_dict['journal_abbrev'] = journalabbrev_
        if 'volume' in dataframe:
            volume_ = str(row['volume'])
            bibs_dict['volume'] = volume_
        if 'issue' in dataframe:
            issue_ = str(row['issue'])
            bibs_dict['issue'] = issue_
        if 'doi' in dataframe:
            doi_ = str(row['doi'])
            bibs_dict['doi'] = doi_
        if 'pages' in dataframe:
            pages_ = str(row['pages'])
            bibs_dict['pages'] = pages_

        bibs.append(bibs_dict)
        # print("bibs_dict")
        # print(bibs_dict)
        # print("string is")
        # print(string)
        terms = tokenize.split(string)


        terms = [w for w in terms if w != '' and
                 w not in stopwords and
                 not exclude.match(w)]

        terms_total += terms

        # df
        for w in set(terms):
            if w in df:
                df[w] += 1
            else:
                df[w] = 1
                w2id[w] = cnt_w
                cnt_w += 1

        author_sequence_pointer = 1
        # author tf
        for author in author_:
            author_doc_dict[author].append((i, author_sequence_pointer)) # label as the n-th author of current article
            author_sequence_pointer += 1
            # author_names.add(author)
            for w in terms:
                author_term_dict[author][w] += 1

        # doc tf
        tf = dict()
        for w in terms:
            if w in tf:
                tf[w] += 1
            else:
                tf[w] = 1
        doc_term_mat.append(tf)

        i += 1

    author_names = list(author_term_dict.keys())
    terms_total = set(terms_total)
    return author_names, doc_term_mat, df, w2id, author_term_dict, author_doc_dict, bibs

def kmeans_authors(data, keywords, n_components, k, n_desc=5):
    print("Kmeans authors...")

    # in case k is smaller than num of docs
    error_r = None
    k = min(k, data.shape[0])
    data = data / norm(data, axis=1)[:, np.newaxis]

    # SVD
    cluster_labels = []
    if n_components == 0:  # no svd
        km = KMeans(init='k-means++', n_clusters=k, n_init=10,
                    random_state=0)
        km.fit(data)

        # set cluster centers to get descriptions
        centers = km.cluster_centers_

    else:  # apply svd then cluster
        svd_model = TruncatedSVD(algorithm="randomized",
                                 n_components=n_components,
                                 random_state=42)
        try:
            svd_model.fit(data)
        except:
            error_r = "Try another search, I couldn't find enough information to analyze."
            return None, None, None, None, error_r

        reduced_data = svd_model.transform(data)

        km = KMeans(init='k-means++', n_clusters=k,
                    n_init=10, random_state=0).fit(reduced_data)

        centers = svd_model.inverse_transform(km.cluster_centers_)

    # get cluster descriptions
    for j, c in enumerate(centers):
        # in case n_desc is greater than num of keywords
        n_desc_ = min(n_desc, len(c)-1)
        i = np.argpartition(-c, n_desc_)[:n_desc_]  # top n (unsorted)
        c_i = c[i]
        i_ = np.argsort(c_i)[::-1]  # sort
        k_i = np.array(keywords)[i]
        labels_ = k_i[i_].tolist()
        labels_ = [i for i in labels_ if len(i) > 3]
        # print("C%d: " % (j+1) + ", ".join(labels_))
        cluster_labels.append(labels_)

    # get 2d coordinates by t-SNE
    tsne = MyTSNE(n_components=2,
                  n_iter=5000, random_state=123)
    cluster_centers = tsne.fit_transform(
        km.cluster_centers_)

    '''
    # get 2d coordinates by svd
    svd_model = TruncatedSVD(n_components=2,
                             random_state=42)
    cluster_centers = svd_model.fit_transform(km.cluster_centers_)
    '''

    # scale to (0,1)
    scaler = preprocessing.MinMaxScaler()
    cluster_centers = scaler.fit_transform(cluster_centers).tolist()

    # scale intermediate data too
    coordinates = scaler.transform(tsne.coordinates.reshape(-1, 2)).\
        reshape(tsne.coordinates.shape).tolist()

    return km.labels_.tolist(), cluster_centers,\
        cluster_labels, coordinates, error_r

def kmeans_doc_doc(doc_term_mat, keywords, org_ids, n_components, k, n_desc=5):
    print("Kmeans doc-doc...")
    error_r = None
    # in case k is smaller than num of doc_term_mat
    k = min(k, doc_term_mat.shape[0])
    # print('pre normalize')
    doc_term_mat = doc_term_mat / norm(doc_term_mat, axis=1)[:, np.newaxis]

    # column dimensionality of the transposed data
    col_dim = doc_term_mat.shape[1]
    # to use TruncatedSVD later, n_components must be strictly less than the # of features, in this case, the dimentionality of columns (documents)
    n_components = min(n_components, col_dim - 1)

    # SVD
    if n_components == 0:  # no svd
        km = KMeans(init='k-means++', n_clusters=k, n_init=10,
                    random_state=0)
        km.fit(doc_term_mat)

        # set cluster centers to get descriptions
        centers = km.cluster_centers_

    else:
        try:
            doc_doc_mat = cosine_similarity(doc_term_mat)
        except:
            error_r = "Problems with cosine similarity matrix!"
            return None, None, None, None, error_r

        try:
            km = KMeans(init='k-means++', n_clusters=k,
                        n_init=10, random_state=0).fit(doc_doc_mat)
        except:
            error_r = "Problems kmeans fitting!"
            return None, None, None, None, error_r


    from collections import defaultdict
    id2members = defaultdict(list)
    for i, m in enumerate(km.labels_):
        bib_id = org_ids[i]
        id2members[int(m)].append(bib_id)

    id2members = dict(sorted(id2members.items(), key=operator.itemgetter(0), reverse=False))  # sort id2members in ascending order by key
    # get 2d coordinates by t-SNE
    tsne = MyTSNE(n_components=2,
                  n_iter=5000, random_state=123)
    cluster_centers = tsne.fit_transform(
        km.cluster_centers_)  # only calculate tsne for the populated_ids

    # scale to (0,1)
    scaler = preprocessing.MinMaxScaler()
    cluster_centers = scaler.fit_transform(cluster_centers).tolist()

    # scale intermediate data too
    coordinates = scaler.transform(tsne.coordinates.reshape(-1, 2)).\
        reshape(tsne.coordinates.shape).tolist()

    return id2members, cluster_centers, coordinates, error_r

def kmeans_author_author(author_term_mat, keywords, org_ids, n_components, k, n_desc=5):
    from collections import defaultdict
    print("Kmeans author-author...")
    error_r = None
    # Ensure k is not greater than the number of rows in author_term_mat
    k = min(k, author_term_mat.shape[0])
    
    # Normalize author_term_mat
    author_term_mat = normalize(author_term_mat, axis=1)
    
    # Column dimensionality of the transposed data
    col_dim = author_term_mat.shape[1]
    
    # To use TruncatedSVD later, n_components must be strictly less than the number of features
    n_components = min(n_components, col_dim - 1)
    
    if n_components == 0:  # No SVD
        km = KMeans(init='k-means++', n_clusters=k, n_init=10, random_state=0)
        km.fit(author_term_mat)
        centers = km.cluster_centers_
    else:
        try:
            author_author_mat = cosine_similarity(author_term_mat)
        except Exception as e:
            error_r = f"Problems with cosine similarity matrix: {e}"
            return None, None, None, None, error_r
        
        try:
            km = KMeans(init='k-means++', n_clusters=k, n_init=10, random_state=0).fit(author_author_mat)
        except Exception as e:
            error_r = f"Problems kmeans fitting: {e}"
            return None, None, None, None, error_r
    
    centers = km.cluster_centers_
    
    cluster_author_indices = []
    for j, c in enumerate(centers):
        # In case n_desc is greater than the number of keywords
        n_desc_ = min(n_desc, len(c) - 1)
        i = np.argpartition(-c, n_desc_)[:n_desc_]  # Top n (unsorted)
        c_i = c[i]
        i_ = np.argsort(c_i)[::-1]  # Sort
        cluster_author_indices.append(i_)
    
    id2authors = defaultdict(list)
    for i, author_ids in enumerate(cluster_author_indices):
        org_author_ids = []
        for aid in author_ids:
            org_author_ids.append(org_ids[aid])
        id2authors[i] = org_author_ids
    
    # Return id2authors, cluster_centers, coordinates, and error_r
    coordinates = km.cluster_centers_  # Assuming you want to return cluster centers as coordinates
    return id2authors, centers, coordinates, error_r


def process_authors_4(dataframe, dataset, stopwords=[]):

    print("Processing authors...")
    author_names, doc_term_mat, df, w2id, author_term_dict, author_docs, bibs = read_df_authors_2(dataframe, dataset, stopwords)
    author_term_mat = list(author_term_dict.values())

    # Remove terms whose df is lower than mindf
    MINDF = 1
    if MINDF > 0:
        inf = []
        for w in df:
            if df[w] <= MINDF:
                inf.append(w)
        for w in inf:
            del df[w]

    # save original matrix and df
    orig_author_term_mat = author_term_mat
    orig_df = df

    author_term_mat, dfr = compute_tfidf(author_term_mat, df, rank=5)

    keywords = output_keywords(len(author_term_mat), dfr, df, p_docs=1.0)

    author_term_mat, org_ids = update(author_term_mat, keywords)

    author_term_mat = convert_sparse(author_term_mat, keywords)
    id2authors, cluster_centers, coordinates, error = \
        kmeans_author_author(author_term_mat, keywords, org_ids, n_components=20, k=10, n_desc=15)



    cluster_relations = []
    cluster_desc = []
    id2members = {}
    for cluster in sorted(id2authors.keys()):
        author_ids = id2authors[cluster]
        authors = []
        for i in author_ids:
            authors.append(author_names[i])

        documents = set()
        for author in authors:
            docs = author_docs[author]
            for doc_tuple in docs:
                (doc_id, author_sequence) = doc_tuple
                if author_sequence == 1 or author_sequence == 2: # first/second author of this doc
                    documents.add(doc_id)

        if not documents: # if so unlucky that no authors in this cluster are listed as first/second author of any articles
            top_n = min(3, len(authors))
            for author in authors[:top_n]: # only for the first few authors in cluster
                docs = author_docs[author]
                most_related_doc = min(docs, key=lambda x:x[1]) # get the article where current author is listed at the highest position compared to his/her other articles
                documents.add(most_related_doc[0])

        authors = authors[:15]
        cluster_desc.append(authors)
        id2members[cluster] = list(documents)
        cluster_relations.append({'authors': authors, 'documents': list(documents)})

    # for k, v in enumerate(bibs):
    #     print(k, v['title'])
    #     print(v['author'])
    #     print('\n')

    # for rel in cluster_relations:
    #     print(rel)
    #     print('\n')

    return orig_author_term_mat, orig_df, df, dfr, keywords, cluster_centers, cluster_desc, coordinates, id2members, bibs, org_ids, author_names, author_docs


# see how many articles in each cluster actually have one of their cluster gene labels in the their title + abstract

def genes_coverage(gene_names, references):
    print('KEYWORDS COVERAGE')
    total = len(references)
    print('cluster labels: ' + str(gene_names))
    print('total amount of article: ' + str(total))
    coverage = 0.0
    for ref in references:
        content = ref['title'] + ref['abstract']
        terms = tokenize.split(content)

        for gene in gene_names:
            if gene in terms:
                print(gene)
                coverage += 1.0
                break

    print('number of articles containing cluster label: ' + str(coverage))

    print('coverage: ' + str(coverage / total))


'''
Compute tfidf and find key terms
'''


def compute_tfidf(doc_term_mat, df, rank=5):
    print("Computing tfidf...")

    # print("Does df contains none?")
    # print(None in df)
    # alternative procedure to generate new doc_term_mat copy, avoding using del keyword
    doc_term_mat_ = []
    for d in doc_term_mat:
        d_ = {}
        for w in d:
            if w in df:
                d_[w] = d[w]
        if d_:
            doc_term_mat_.append(d_)
    # doc_term_mat_ = copy.deepcopy(doc_term_mat) # make copy

    dfr = dict()  # df considering only top R terms per document
    for i in range(len(doc_term_mat_)):
        for w in doc_term_mat_[i]:
            doc_term_mat_[i][w] *= math.log(len(doc_term_mat_)/df[w])
        # for w in doc_term_mat[i]:
        #     # delete low DF term
        #     if w not in df:
        #         del doc_term_mat_[i][w]
        #         continue
        #     # tfidf
        #     doc_term_mat_[i][w] *= math.log(len(doc_term_mat)/df[w])

        # Ignore if rank = 0
        if rank > 0:
            # Sort and extract top R terms for this document
            terms_sorted = sorted(doc_term_mat_[i].items(), reverse=True,
                                  key=operator.itemgetter(1))
            top_r = terms_sorted[:rank]

            # Count new df for only top R
            for w, _ in top_r:
                if w in dfr:
                    dfr[w] += 1
                else:
                    dfr[w] = 1

    return doc_term_mat_, dfr


'''
Output matrix
'''


def output_matrix(csv_dir, filename, doc_term_mat, vocab):
    print("Outputting matrix...")
    vocab = list(vocab)
    vocab.sort()
    with open(os.path.join(csv_dir, filename), 'wb') as f:
        # header
        f.write((','.join(vocab) + '\n').encode('utf8'))
        # values
        for i, d in enumerate(doc_term_mat):
            out = ''
            for w in vocab:
                if w in d:
                    out += "{:.5},".format(d[w])
                else:
                    out += "0,"
            f.write((out.rstrip(',') + '\n').encode('utf8'))
            if i % int(len(doc_term_mat)/20) == 0:
                print("%d%% finished..." % (i/len(doc_term_mat)*100))


'''
Sort and output results (discovered keywords)
'''


def output_keywords(num_docs, dfr, df, p_docs=0.5, html=False):
    print("Outputting keywords...")
    min_dfr = num_docs * p_docs / 100
    terms_sorted = sorted(dfr.items(), reverse=True,
                          key=operator.itemgetter(1))
    # if not html:
    #     print("term\tDF\tDFR")

    keywords = []
    dfs = []
    dfrs = []
    for w, v in terms_sorted:
        if v < min_dfr:
            break
        if w not in df:
            continue

        keywords.append(w)
        dfs.append(df[w])
        dfrs.append(dfr[w])

        # if not html:
        #     print("%s\t%d\t%d" % (w, df[w], dfr[w]))

    if html:
        d = pd.DataFrame({"Term": keywords, "DF": dfs, "DFR": dfrs},
                         columns=["Term", "DF", "DFR"])
        display(HTML(d.to_html(index=False)))

    return keywords


'''
Normalize matrix (not used)
'''


def normalize(mat, axis='document'):
    print("Normalizing matrix...")
    if axis == 'document':
        for i in range(len(mat)):
            norm = 0
            for w in mat[i]:
                norm += mat[i][w] * mat[i][w]
            norm = math.sqrt(norm)
            for w in mat[i]:
                mat[i][w] /= norm
    elif axis == 'term':
        # not implemented
        pass
    return mat


'''
Maximin (core)
'''


def maximin_core(docs, m, what_to_cluster="document",
                 keywords=[], theta=0.9, verbose=False):
    print("Maximin clustering...")

    # compute similarity matrix
    sim = np.array(m.dot(m.transpose()))
    np.fill_diagonal(sim, -2)  # need to set smaller than -1
    centroids = []
    candidates = list(range(sim.shape[0]))

    # pick the first centroid
    random.seed(10)
    centroids.append(candidates.pop(
        random.randint(0, sim.shape[0]-1)))

    # Find next centroid iteratively
    while True:
        sim_max = sim[centroids, :].max(axis=0)  # take max as this is
        # similarity
        sim_max[centroids] = 2  # need to be greater than 1
        maximin_id = sim_max.argmin()
        maximin = 1 - sim_max.min()  # make similarity to distance

        if maximin > theta:
            centroids.append(candidates.pop(
                candidates.index(maximin_id)))
        else:
            break

    # Results
    print("%d clusters generated" % len(centroids))
    print(centroids)
    print()

    # Show clusters
    if what_to_cluster == "document":
        np.fill_diagonal(sim, 1)  # set back to 1
        membership = sim[centroids, :].argmax(axis=0).tolist()
        if verbose:
            for i, id in enumerate(centroids):
                print('Doc %d cluster (%d members):' %
                      (id, len([x for x in membership if x == i])))
                print(docs[id])
                print()

        # renumber membership
        m2id = {m: i for i, m in enumerate(set(membership))}
        membership = [m2id[m] for m in membership]

    else:
        np.fill_diagonal(sim, 1)  # set back to 1
        membership = sim[centroids, :].argmax(axis=0)
        print(membership)

        if verbose:
            keywords_ = np.asarray(keywords)
            for i, id in enumerate(centroids):
                print('Term %d \"%s\" cluster (%d members):' %
                      (id, keywords[id], (membership == i).sum()))
                print(keywords_[membership == i])
                print()

        membership = membership.tolist()

    # compute Silhouette Coefficient. can't use sim
    # since diagonals were changed.
    if len(set(membership)) < 2:
        sc = float('nan')
    else:
        sc = metrics.silhouette_score(m, membership,
                                      metric='cosine')

    return centroids, membership, sim, sc


'''
Maximin clustering wrapper
'''


def maximin(csv_dir, docs, file_sim, cluster, keywords,
            true_labels=[],
            theta=0.9, n_components=0, verbose=True):
    print("Maximin clustering...")
    # add ids to keywords
    keywords.sort()
    w2id = {c: i for i, c in enumerate(keywords)}

    # Convert to scipy matrix for faster calculation
    data = []
    row_idx = []
    col_idx = []
    for i in range(len(docs)):
        data += docs[i].values()
        col_idx += [w2id[w] for w in docs[i].keys()]
        row_idx += [i] * len(docs[i])

    m = csr_matrix((data, (row_idx, col_idx)),
                   (len(docs), len(keywords)))

    # Compute similarity matrix (cosine similarity)
    if cluster == "document":
        pass
    else:  # assume 'term' if not 'document'
        m = m.transpose()

    # normalization
    m = m / norm(m, axis=1)[:, np.newaxis]

    # SVD
    cluster_labels = []
    sct = 0
    if n_components == 0:  # no svd
        centroids, membership, sim, sc = \
            maximin_core(docs, m, cluster, keywords, theta, verbose)
    else:  # apply svd then cluster
        svd_model = TruncatedSVD(n_components=n_components,
                                 random_state=42)
        svd_model.fit(m)
        reduced_m = svd_model.transform(m)
        '''
        # normalization
        reduced_m = reduced_m / \
            np.linalg.norm(reduced_m, axis=1)[:,np.newaxis]
        '''
        centroids, membership, sim, sc = \
            maximin_core(docs, reduced_m, cluster,
                         keywords, theta, verbose)

    '''
    # create cluster labels by inverse-transforming
    # cluster centers and take 5 words with highest values
    centers = m[centroids]
    for c in centers:
        i = np.argpartition(c, -5)[-5:]
        labels_ = np.array(keywords)[i].tolist()
        print(labels_)
        cluster_labels.append(labels_)
    '''

    if n_components > 0:
        print("Formed %d clusters after reducing "
              "to %d dimensions." % (len(set(membership)),
                                     n_components))
    else:
        print("Formed %d clusters w/o SVD." %
              len(set(membership)))

    # Write similarity matrix to csv file
    if file_sim:
        print("Writing similarity matrix to file...")

        with open(os.path.join(csv_dir, file_sim), 'wb') as f:
            # header
            if cluster == "document":
                f.write((",".join(
                    [str(i) for i in range(len(docs))]))
                    .encode('utf8'))
            else:
                f.write((",".join(keywords)).encode('utf8'))
            f.write("\n".encode('utf8'))
            # matrix
            for i in range(sim.shape[0]):
                row = sim[i, :].tolist()
                f.write((','.join(
                    ["{}".format(x) for x in row]))
                    .encode('utf8'))
                f.write("\n".encode('utf8'))
                if i % int(len(docs)/20) == 0:  # progress
                    print("%d%% finished..." % (i/len(docs)*100))

    return centroids, membership, sim, sc, sct


'''
Visualize similarity network
'''


def visualize_network(sim, labels, group):
    print("Visualizing network...")

    np.fill_diagonal(sim, 0)
    G = ig.Graph.Adjacency((sim > .1).tolist())
    G.es['weight'] = sim
    G.vs['label'] = labels

    Edges = [e.tuple for e in G.es]

    # labels = [range(sim.shape[0])]
    # group = [0] * sim.shape[0]

    layt = G.layout('kk', dim=3)

    N = sim.shape[0]
    Xn = [layt[k][0] for k in range(N)]  # x-coordinates of nodes
    Yn = [layt[k][1] for k in range(N)]  # y-coordinates
    Zn = [layt[k][2] for k in range(N)]  # z-coordinates
    Xe = []
    Ye = []
    Ze = []

    for e in Edges:
        # x-coordinates of edge ends
        Xe += [layt[e[0]][0], layt[e[1]][0], None]
        Ye += [layt[e[0]][1], layt[e[1]][1], None]
        Ze += [layt[e[0]][2], layt[e[1]][2], None]

    trace1 = go.Scatter3d(x=Xe,
                          y=Ye,
                          z=Ze,
                          mode='lines',
                          line=dict(color='rgb(125,125,125)', width=1),
                          hoverinfo='none'
                          )

    trace2 = go.Scatter3d(x=Xn,
                          y=Yn,
                          z=Zn,
                          mode='markers+text',
                          name='terms',
                          marker=dict(symbol='circle',
                                      size=6,
                                      color=group,
                                      colorscale='Viridis',
                                      line=dict(
                                          color='rgb(50,50,50)', width=0.5)
                                      ),
                          text=labels,
                          textposition='top center',
                          hoverinfo='x+y+z'
                          )

    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    layout = go.Layout(
        title="Network of the discovered keywords (3D visualization)",
        width=1000,
        height=1000,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
    )

    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)

    py.offline.iplot(fig, filename='network')


'''
Create new matrix with given keywords
'''


def update(doc_term_mat, keywords, prev_ids=None):
    print("Updating matrix...")

    if prev_ids == None:
        prev_ids = range(len(doc_term_mat))

    doc_term_mat_new = []
    org_ids = []

    for i, d in enumerate(doc_term_mat):
        h = dict()
        for w in keywords:
            if w in d:
                h[w] = d[w]
                # replace 0 with tiny value to avoid zero division when calculating norm in the future
                if h[w] == 0.0:
                    h[w] = 0.001
        if len(h) > 0:  # exclude doc w/ no terms
            doc_term_mat_new.append(h)
            org_ids.append(prev_ids[i])

    return doc_term_mat_new, org_ids


'''
Delete low-df words
'''


def del_lowdf(df, mindf=1):
    print("Deleting low-df words...")
    inf = []
    for w in df:
        if df[w] <= mindf:
            inf.append(w)
    for w in inf:
        del df[w]

    print("%d terms were removed" % len(inf))


'''
main
'''
if __name__ == "__main__":
    main()
