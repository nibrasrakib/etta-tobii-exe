from flask_login import current_user
import visual_library_plos as vl
import simplejson
from urllib.parse import urlencode
from urllib.request import urlopen
from io import StringIO
from contextlib import closing
from flask_session import Session
from flask import (
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    g,
    flash,
    session,
    jsonify,
    Response,
    json,
)
from sklearn import preprocessing
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
from app import app, tokenize, exclude, stopwords, MINDF
from db.db_get_data import get_dc_data

####
# Tobii related imports
from app import socketio
from flask_socketio import emit
import tobii_research as tr
import threading
####


import math

import os
import time

time.clock = time.time
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
import json
# Import requests library
import requests
from tqdm import tqdm

import pysolr
from elasticsearch import Elasticsearch
from openai import OpenAI
from generate_summary import generate_summary
from cluster_postgreSQL import get_date_summary

from app import db # for database connection

import logging
# Set logging level to suppress debug/info logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

es = Elasticsearch()
# import spacy


# for flask

# from flask.ext.uploads import UploadSet, configure_uploads, TEXT

# for connecting to PLOS server

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


google_api_mapping = {
    "GoogleAPI": "googleAPI",
    "GoogleAPI2": "googleAPI2",
    "GoogleAPI3": "googleAPI3",
    "GoogleAPI4": "googleAPI4",
    "GoogleAPI5": "googleAPI5",
}


def google_cluster(dataset, num_cls):
    print("google_cluster")
    import importlib

    module_name = google_api_mapping[dataset]
    googleAPImodule = importlib.import_module(module_name)
    # print("start_date")
    # print(start_date)
    num_cls = int(num_cls)
    # if no query is entered exit the function and redirect url to home
    if q == "":
        return redirect(url_for("home"))
    data, error = googleAPImodule.retrieve(q)
    if data.empty:
        return redirect(url_for("home"))
    if error == None:
        doc_term_mat, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
        return doc_term_mat, df, w2id, bibs
    else:
        # #flask(error)
        return redirect(url_for("home"))


@app.route("/")
def home():
    print("home")
    # if 'gene_set' not in session:
    #     session['gene_set'] = vl.load_gene_data()
    env = request.environ

    zoom_cred = os.environ.get("ZOOM_CREDENTIAL", "")
    zoom_code = request.args.get("code", None)
    if zoom_code and zoom_cred:
        redirect_uri = "https://pattie.unc.edu/users-alpha"
        url = f"https://zoom.us/oauth/token?grant_type=authorization_code&code={zoom_code}&redirect_uri={redirect_uri}"
        headers = {"Authorization": f"Basic {zoom_cred}"}

        response = requests.post(url, headers=headers)
        r_json = response.json()
        access_token = r_json["access_token"]

        endpoint = "https://api.zoom.us/v2/users/me/recordings"
        auth_headers = {"Authorization": f"Bearer {access_token}"}

        endpoint_response = requests.get(endpoint, headers=auth_headers)
        endpoint_json = endpoint_response.json()
        print(endpoint_json)

    # set username if onyen is detected, make a new upload directory for new user
    if "eppn" in env:
        username = env["eppn"].split("@")[0]
        login = "true"
        loginType = "onyen"
        path = os.path.join(app.config["UPLOAD_FOLDER"], username)
        if not os.path.exists(path):
            os.mkdir(path)
    # if no onyen login, check if there is general login
    elif current_user.is_authenticated:
        username = current_user.email
        login = "true"
        loginType = "general"
        path = os.path.join(app.config["UPLOAD_FOLDER"], username)
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        username = "username"
        login = "false"
        loginType = "none"
        path = app.config["UPLOAD_FOLDER"]

    uploaded_files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if ".txt" in file or ".csv" in file:
                uploaded_files.append(file)
    # uploaded_files.append("Upload...")
    field = "All fields"
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    # Print all arguments being passed to the home page 
    print("Arguments passed to home page: ")
    print("login: ", login, "loginType: ", loginType, "username: ", username, "field: ", field, "uploaded_files: ", uploaded_files, "start_date: ", yesterday, "end_date: ", today)
    # Print is_authenicated
    print("is_authenticated: ", current_user.is_authenticated)
    return render_template(
        "index.html",
        login=login,
        loginType=loginType,
        username=username,
        field=field,
        uploaded_files=uploaded_files,
        start_date=yesterday,
        end_date=today,
        is_authenticated=current_user.is_authenticated,
    )

@app.route("/calibration", methods=["GET"])
def calibration():
    query = request.args.get("query", "")
    num_cls = request.args.get("num_cls", 10)
    # Retrieve sessionData from Flask session
    sessionData = session.get("sessionData", {})

    # Pass sessionData and other parameters to the calibration template
    return render_template("calibration.html", query=query, num_cls=num_cls, sessionData=sessionData)


@app.route("/_back", methods=["POST", "GET"])
def back():
    print("back")
    response_time = "0"
    send_time = time.clock()

    # Get sessionStorage data
    sessionData = request.values.get("sessionData")
    sessionData = json.loads(sessionData)
    # Get current state id
    state = sessionData["state"]
    # print("back state: " + str(state))

    ids = []
    # Error check and get the chosen ids; if at the first stage, keep the current data
    if state == 0:
        state = state + 1
        stage_alert = "first"
        ids = sessionData["chosen_0"]
    elif state == 1:
        stage_alert = "about to be the first"
        # get the chosen ids
        ids = sessionData["chosen_" + str(state - 1)]
    else:
        stage_alert = ""
        # get the chosen ids
        ids = sessionData["chosen_" + str(state - 1)]

    id2members = sessionData["id2members_" + str(state - 1)]
    id2freq = {x: len(id2members[x]) for x in id2members}

    # making the chosen list
    chosen = []
    for i in id2members:
        if str(i) in ids:
            chosen += [1]
        else:
            chosen += [0]

    # color
    val = [0.6] * len(id2freq)

    # Update state id
    sessionData["state"] = state - 1

    receive_time = time.clock()
    response_time = str(round(receive_time - send_time, 3))

    # Prepare data to send
    cls = {
        "stage_alert": stage_alert,
        "id2freq": id2freq,
        "desc": sessionData["cluster_desc_" + str(state - 1)],
        "xy": sessionData["xy_" + str(state - 1)],
        "hue": sessionData["hue_" + str(state - 1)],
        "satr": sessionData["satr_" + str(state - 1)],
        "val": val,
        "bibs": session["bibs_" + str(state - 1)],
        "id2members": id2members,
        "sources": session["sources"],
        "chosen": chosen,
        "edges": sessionData["edges_" + str(state - 1)],
    }

    return jsonify(cls=cls)

@app.route("/_re_cluster", methods=["POST", "GET"])
def re_cluster():
    print("re_cluster")

    response_time = "0"
    send_time = time.clock()

    # Get sessionStorage data
    # print('------------------------SESSION DATA --------------------------')
    sessionData = request.values.get("sessionData")
    sessionData = json.loads(sessionData)
    # print(sessionData)
    # Get session data
    state = sessionData["state"]
    print("reclustering---->  state in re_cluster initially: ", state)
    num_cls = sessionData["num_cls"]
    dataset = sessionData["dataset"]
    print("reclustering ----> dataset: ", dataset, "num_cls: ", num_cls)
    # Get cluster ids
    ids = sessionData["ids_" + str(state)]
    # print("ids: ", ids)

    # Get doc ids (in selected clusters)
    id2members = sessionData["id2members_" + str(state)]
    doc_ids = []
    for id in ids:
        doc_ids += id2members[id]

    doc_ids = list(set(doc_ids))  # remove duplicate doc ids
    entity = session["entity"]
    
    print("reclustering ----> entity: ", entity)
    print("reclustering ----> doc_ids: ", doc_ids)

    if dataset == "Experts":
        num_cls = get_num_cls_for_experts(len(doc_ids))

        doc_org = sessionData["docs_org"]
        org_ids = sessionData["org_ids_" + str(state)]
        doc_term_mat = []
        org_ids_ = []  # original document ids
        for id in doc_ids:
            id_ = id
            org_ids_.append(id_)
            doc_term_mat.append(doc_org[id_])

        doc_term_mat, dfr = vl.compute_tfidf(
            doc_term_mat, sessionData["df_org"], rank=5
        )

        keywords = vl.output_keywords(
            len(doc_term_mat), dfr, sessionData["df_org"], p_docs=1.0
        )

        doc_term_mat, org_ids_ = vl.update(doc_term_mat, keywords, org_ids_)
        sessionData["org_ids_" + str(state + 1)] = org_ids_

        bibs = session["bibs_0"]
        bibs_new = []
        for id in org_ids_:
            # print(id, bibs[id])
            bibs_new.append(bibs[id])
        session["bibs_" + str(state + 1)] = bibs_new

        doc_term_mat = vl.convert_sparse(doc_term_mat, keywords)
        if entity == "experts":
            id2members, cluster_centers, coordinates, error = vl.kmeans_doc_doc(
                doc_term_mat, keywords, org_ids, n_components=20, k=num_cls, n_desc=15
            )

            cluster_desc = []
            for i, m in id2members.items():
                desc = [
                    bibs[j]["title"] for j in m[:15]
                ]  # get the first 15 expert names as cluster labels
                cluster_desc.append(desc)
        else:  # entity is 'topics'
            id2members, cluster_centers, cluster_desc, coordinates, error = vl.kmeans(
                doc_term_mat, keywords, org_ids, n_components=20, k=num_cls, n_desc=15
            )

    elif entity == "authors":
        # Get doc data
        author_org = sessionData["docs_org"]
        org_ids = sessionData["org_ids_" + str(state)]
        author_term_mat = []
        org_ids_ = []  # original document ids
        for id in doc_ids:
            # id_ = org_ids[id]
            id_ = id
            org_ids_.append(id_)
            author_term_mat.append(author_org[id_])

        # Redo feature selection
        # print("Re-computing TFIDF and finding key terms...")
        author_term_mat, dfr = vl.compute_tfidf(
            author_term_mat, sessionData["df_org"], rank=5
        )

        # Sort and output results (discovered keywords)
        keywords = vl.output_keywords(
            len(author_term_mat), dfr, sessionData["df_org"], p_docs=0.5
        )

        # Create new matrix with the keywords
        author_term_mat, org_ids_ = vl.update(author_term_mat, keywords, org_ids_)
        sessionData["org_ids_" + str(state + 1)] = org_ids_

        bibs = session["bibs_0"]
        bibs_new = []
        for id in org_ids_:
            print(id, bibs[id])
            bibs_new.append(bibs[id])
        session["bibs_" + str(state + 1)] = bibs_new

        author_term_mat = vl.convert_sparse(author_term_mat, keywords)
        id2authors, cluster_centers, coordinates, error = vl.kmeans_author_author(
            author_term_mat, keywords, org_ids, n_components=20, k=10, n_desc=15
        )

        author_names = session["author_names"]
        author_docs = session["author_docs"]
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
                    if (
                        author_sequence == 1 or author_sequence == 2
                    ):  # first/second author of this doc
                        documents.add(doc_id)

            if (
                not documents
            ):  # if so unlucky that no authors in this cluster are listed as first/second author of any articles
                top_n = min(3, len(authors))
                for author in authors[
                    :top_n
                ]:  # only for the first few authors in cluster
                    docs = author_docs[author]
                    most_related_doc = min(
                        docs, key=lambda x: x[1]
                    )  # get the article where current author is listed at the highest position compared to his/her other articles
                    documents.add(most_related_doc[0])

            authors = authors[:15]
            cluster_desc.append(authors)
            id2members[cluster] = list(documents)

    else:
        # Get doc data
        doc_org = sessionData["docs_org"]
        org_ids = sessionData["org_ids_" + str(state)]
        doc_term_mat = []
        org_ids_ = []  # original document ids
        for id in doc_ids:
            # id_ = org_ids[id]
            id_ = id
            org_ids_.append(id_)
            doc_term_mat.append(doc_org[id_])

        # Redo feature selection
        # print("Re-computing TFIDF and finding key terms...")
        doc_term_mat, dfr = vl.compute_tfidf(
            doc_term_mat, sessionData["df_org"], rank=5
        )

        # Sort and output results (discovered keywords)
        keywords = vl.output_keywords(
            len(doc_term_mat), dfr, sessionData["df_org"], p_docs=0.5
        )

        # Create new matrix with the keywords
        doc_term_mat, org_ids_ = vl.update(doc_term_mat, keywords, org_ids_)
        sessionData["org_ids_" + str(state + 1)] = org_ids_

        # Prepare bibliographies if num of documents is small

        # if len(org_ids_) < 50:
        bibs = session["bibs_0"]
        bibs_new = []
        for id in org_ids_:
            # print(id, bibs[id])
            bibs_new.append(bibs[id])
        session["bibs_" + str(state + 1)] = bibs_new

        # Convert to sparse matrix

        doc_term_mat = vl.convert_sparse(doc_term_mat, keywords)
        
        if num_cls is None:
            num_cls = optimal_number_of_clusters(doc_term_mat, max_clusters=10)
            print("reclustering ----> optimal number of clusters: ", num_cls)
            
            
        # Re-cluster
        if dataset == "NYTIMES":
            id2members, cluster_centers, cluster_desc, coordinates, error = vl.kmeans(
                doc_term_mat, keywords, org_ids_, n_components=7, k=num_cls, n_desc=15
            )
        else:
            # print(dataset)  # newly add
            id2members, cluster_centers, cluster_desc, coordinates, error = vl.kmeans(
                doc_term_mat, keywords, org_ids_, n_components=10, k=num_cls, n_desc=25
            )

    id2freq = {x: len(id2members[x]) for x in id2members}

    # Get cluster colors
    cc = np.array(cluster_centers) - 0.5
    hue = ((np.arctan2(cc[:, 1], cc[:, 0]) / np.pi + 1) * 180).astype(int).tolist()
    satr = (np.linalg.norm(cc, axis=1) / math.sqrt(2) * 2).tolist()
    val = [0.6] * len(id2freq)
    print("Obtain cluster colors")

    # create adjacency matrix that will be used for network edges
    centroid_matrix = pd.DataFrame.from_records(cluster_centers)
    centroid_distances = pd.DataFrame(
        squareform(pdist(centroid_matrix, metric="euclidean")),
        columns=list(id2members.keys()),
        index=list(id2members.keys()),
    ).to_dict()

    # create network data structure
    # print('\nEUCLIDEAN DISTANCES\n----------')
    edges = []
    for i in id2members:
        for k, v in centroid_distances[i].items():
            # print(i, ' to ', k, ' = ', v)
            if i != k:
                if v > 0.75:
                    v = "Distant"
                    edge = {
                        "clusterID": i,
                        "source": cluster_centers[i],
                        "target": cluster_centers[k],
                        "distance": v,
                    }
                    edges.append(edge)
                elif v <= 0.75 and v > 0.50:
                    v = "Similar"
                    edge = {
                        "clusterID": i,
                        "source": cluster_centers[i],
                        "target": cluster_centers[k],
                        "distance": v,
                    }
                    edges.append(edge)
                elif v <= 0.50:
                    v = "Very Similar"
                    edge = {
                        "clusterID": i,
                        "source": cluster_centers[i],
                        "target": cluster_centers[k],
                        "distance": v,
                    }
                    edges.append(edge)
    # print(edges)

    print('\n reclustering ----> BIBLIOGRAPHY\n----------')
    sources = []
    if dataset == "PubMedAPI":
        from collections import Counter
        
        for cluster_id in sorted(id2members.keys()):
            concepts = cluster_desc[cluster_id]
            references = []
            for bib_id in id2members[cluster_id]:
                references.append(bibs[bib_id])
            for paper in references:
                if type(paper["author"]) != str and None in paper["author"]:
                    paper["author"] = [i for i in paper["author"] if i]
            bibliography = {"concepts": concepts, "references": references}
            sources.append(bibliography)

    
    elif dataset == "PostgreSQL":
        from collections import Counter

        for cluster_id in sorted(id2members.keys()):
            concepts = cluster_desc[cluster_id]
            references = []
            print(f'--------{cluster_id}---------')
            for bib_id in id2members[cluster_id]:
                references.append(bibs[bib_id])
            print(references)
            print('-----------------')
            for paper in references:
                if type(paper["author"]) != str and None in paper["author"]:
                    paper["author"] = [i for i in paper["author"] if i]
                    print(paper["author"])
            # Add 'meshHeadings': [] to each reference in references
            for ref in references:
                ref['meshHeadings'] = []
            bibliography = {"concepts": concepts, "references": references}
            meshTerms = []
            for ref in references:
                meshHeadingList = ref.get("meshHeadings")  # ref["meshHeadings"]
                for mesh in meshHeadingList:
                    meshTerms.append(mesh["descriptor"])
            id2meshTerms[cluster_id] = str(dict(Counter(meshTerms).most_common(20)))
            sources.append(bibliography)
            
    elif dataset == "rss_feed":
        from collections import Counter

        for cluster_id in sorted(id2members.keys()):
            concepts = cluster_desc[cluster_id]
            references = []
            print(f'--------{cluster_id}---------')
            for bib_id in id2members[cluster_id]:
                references.append(bibs[bib_id])
            # print(references)
            print('-----------------')
            for ref in references:
                ref['meshHeadings'] = []
            bibliography = {"concepts": concepts, "references": references}
            sources.append(bibliography)

    
    # Convert NumPy arrays to lists
    def convert_ndarray_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_ndarray_to_list(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
        else:
            return obj
    
    cluster_summary = {}
    cluster_assigned = {}
    cluster_title = {}
    for cluster_id in sorted(id2members.keys()):
        publications = id2members[cluster_id]
        pub_texts = "\n".join([f"@article{{{bib_id}, title={bibs[bib_id]['title']}}}" for bib_id in publications])
        # user_content = f"Clustered Publications:\n\n{pub_texts}\n\nAssigned Cluster: '{cluster_desc[cluster_id]}'"
        user_content = f"Clustered Publications:\n\n{pub_texts}'"
        # print(f"Cluster {cluster_id} \n\n User Content: {user_content}")
        print(f"Cluster {cluster_id} \n\n User Content: {user_content}\n\n")
        summary = generate_summary(user_content)
        # Split the string into parts based on the numbering
        print("Preprocessed Summary: ", summary, "\n\n")
        parts = summary.split("2. ")

        summary_title = parts[0].replace("1. ", "").strip()
        summary_val = parts[1].strip()
        print(f"Cluster {cluster_id}, Assigned Cluster: '{cluster_desc[cluster_id]}'\n\nSummary: {summary_val}\n\nTitle: {summary_title}\n\n")
        if 'N/A' in summary_val or 'N/A' in summary_title:
            summary_val = "No summary available due to insufficient paper content"
            summary_title = "No title available"
        cluster_summary[cluster_id] = summary_val
        cluster_assigned[cluster_id] = cluster_desc[cluster_id]
        # If summary title is a sentence, i want it to be a list of words
        summary_title = summary_title.split()
        cluster_title[cluster_id] = summary_title
    cluster_summary_list = [[value] for key, value in cluster_summary.items()]
    cluster_title_list = [[value] for key, value in cluster_title.items()]
    
    # Store session data
    def unpack_list(lst):
        return [item for sublist in lst for item in sublist]
    
    print("recustering ----> updated state to store session data: ", state+1)
    sessionData["id2members_" + str(state + 1)] = id2members
    sessionData["cluster_desc_" + str(state + 1)] = unpack_list(convert_ndarray_to_list(cluster_title_list))
    sessionData["cluster_summary_" + str(state + 1)] = convert_ndarray_to_list(cluster_summary_list) # Check this later
    sessionData["xy_" + str(state + 1)] = cluster_centers
    sessionData["hue_" + str(state + 1)] = hue
    sessionData["satr_" + str(state + 1)] = satr
    sessionData["chosen_" + str(state)] = ids
    sessionData["edges_" + str(state + 1)] = edges
    # sessionData['sources_' + str(state+1)] = sources
    sessionData["state"] = state + 1

    receive_time = time.clock()
    response_time = str(round(receive_time - send_time, 3))
    print("reclustering ----> cluster_title_list: ", convert_ndarray_to_list(cluster_title_list))
    # Print after [[[...]]] to [[...]] for cluster_title_list by unpacking the list
    
    print("reclustering ----> cluster_title_list unpacked: ", unpack_list(convert_ndarray_to_list(cluster_title_list)))

    cls = {
        "id2freq": id2freq,
        "desc": unpack_list(convert_ndarray_to_list(cluster_title_list)),
        "summary": convert_ndarray_to_list(cluster_summary_list),
        "xy": cluster_centers,
        "xy_inter": coordinates,
        "edges": edges,
        "hue": hue,
        "satr": satr,
        "val": val,
        "bibs": bibs_new,
        "id2members": id2members,
        "sources": sources,
        "response_time": response_time,
        "ear_date": sessionData["ear_date"],
        "lat_date": sessionData["lat_date"],
        "last_up": sessionData["last_up"],
    }
    print("cluster_centers: ", cluster_centers)
    print("state in re_cluster finally", state+1) 
    print("re_cluster done")
    print("Keys of sessionData: ", sessionData.keys())
    return jsonify(cls=cls, newSessionData=sessionData)


"""
Search and cluster the results
"""

entity_search_mapping = {
    "genes": "genetics[subheading]",
    "drugs": "DE[subheading]",
    "diseases": "complications[subheading]",
    "treatment": "diagnosis[subheading] OR therapy[subheading]",
    "pharmacology": "pharmacology[subheading]",
    "epidemiology": "epidemiology[subheading]",
    "anthropology": "ethnology[subheading]",
    "methods": "methods[subheading]",
    "education": "education[subheading]",
    "economics": "economics[subheading]",
    "history": "history[subheading]",
    "trends": "trends[subheading]",
    "africa": "Africa[Mesh]",
    "americas": "Americas[Mesh]",
    "antarctic_regions": '"Antarctic Regions"[Mesh]',
    "arctic_regions": '"Arctic Regions"[Mesh]',
    "asia": "Asia[Mesh]",
    "europe": "Europe[Mesh]",
}


def get_decorated_query(raw_query, entity):
    if entity == "keywords" or entity == "authors":
        return raw_query
    else:
        decoration = entity_search_mapping[entity]
        return raw_query + " AND " + decoration


def get_num_cls_for_experts(num_docs):
    if num_docs >= 16:
        return 10
    elif num_docs >= 12:
        return 8
    elif num_docs >= 10:
        return 6
    elif num_docs >= 8:
        return 5
    else:
        return 3


def get_num_cls_for_digi(num_docs):
    if num_docs >= 16:
        return 10
    elif num_docs >= 12:
        return 8
    elif num_docs >= 10:
        return 6
    elif num_docs >= 8:
        return 5
    else:
        return 3
    
@app.route("/results", methods=["GET"])
def results():
    # Retrieve sessionData from Flask session or POST request
    sessionData = session.get("sessionData", {})
    results_data = session.get("results_data", {})
    # Print results_data
    # print("results_data: ", results_data)
    login = results_data.get("login", "false")
    loginType = results_data.get("loginType", "none")
    username = results_data.get("username", "username")
    query = results_data.get("query", "")
    num_cls = results_data.get("num_cls", 10)
    decoratedQuery = results_data.get("decoratedQuery", "")
    error = results_data.get("error", None)
    keywords = results_data.get("keywords", "")
    df = results_data.get("df", "")
    dfr = results_data.get("dfr", "")
    cluster_desc = results_data.get("cluster_desc", "")
    cluster_summary = results_data.get("cluster_summary", "")
    xy = results_data.get("xy", "")
    edges = results_data.get("edges", "")
    id2freq = results_data.get("id2freq", "")
    xy_inter = results_data.get("xy_inter", "")
    hue = results_data.get("hue", "")
    satr = results_data.get("satr", "")
    val = results_data.get("val", "")
    id2members = results_data.get("id2members", "")
    sources = results_data.get("sources", "")
    dataset = results_data.get("dataset", "")
    response_time = results_data.get("response_time", "")
    path = results_data.get("path", "")
    ear_date = results_data.get("ear_date", "")
    lat_date = results_data.get("lat_date", "")
    last_up = results_data.get("last_up", "")
    print("Arguments passed to results page: ")
    # Print all arguments being passed to the results page
    print("login: ", login, "loginType: ", loginType, "username: ", 
          username, "query: ", query, "num_cls: ", num_cls,
          "dataset: ", dataset, "response_time: ", 
          response_time, "ear_date: ", ear_date, "lat_date: ", lat_date, "last_up: ", last_up)

    # path = sessionData.get("path", "/some/default/path")

    return render_template(
        "results.html",
        query=query,
        num_cls=num_cls,
        sessionData=sessionData,
        path=path, # Pass the path variable
        login=login,
        loginType=loginType,
        username=username,
        decoratedQuery=decoratedQuery,
        error=error,
        keywords=keywords,
        df=df,
        dfr=dfr,
        cluster_desc=cluster_desc,
        cluster_summary=cluster_summary,
        xy=xy,
        edges=edges,
        id2freq=id2freq,
        xy_inter=xy_inter,
        hue=hue,
        satr=satr,
        val=val,
        id2members=id2members,
        sources=sources,
        dataset=dataset,
        response_time=response_time,
        ear_date=ear_date,
        lat_date=lat_date,
        last_up=last_up,
    )


@app.route("/cluster", methods=["POST", "GET"])
def cluster():
    print("cluster")
    es = Elasticsearch(HOST="http://localhost", PORT=5000)
    es = Elasticsearch()
    env = request.environ

    if "eppn" in env:
        username = env["eppn"].split("@")[0]
        login = "true"
        loginType = "onyen"
        path = os.path.join(app.config["UPLOAD_FOLDER"], username)
    elif current_user.is_authenticated:
        username = current_user.email
        login = "true"
        loginType = "general"
        path = os.path.join(app.config["UPLOAD_FOLDER"], username)
    else:
        username = "username"
        login = "false"
        loginType = "none"
        path = app.config["UPLOAD_FOLDER"]

    response_time = "0"
    state = 0
    send_time = time.clock()
    error = None
    # dataset = request.args.get("dataset_opt", "")
    dataset = "rss_feed"

    field = "All fields"
    raw_query = request.args.get("query", "")
    entity = request.args.get("entity", "")
    # q = get_decorated_query(raw_query, entity)
    q = raw_query
    # print(q)
    num_cls = int(request.args.get("num_cls", 10))
    
    print("dataset: ", dataset, "entity: ", entity, "num_cls: ", num_cls, "q: ", q, "raw_query: ", raw_query)

    import cluster_news

    if dataset == "Placeholder":
        return redirect(url_for("home"))

    # dynamic dataset
    elif dataset == "NewsAPI":
        # start_date = request.form['start_date']
        # end_date = request.form['end_date']
        start_date = "2021-03-16"
        end_date = "2021-03-17"
        # print('try')
        num_cls = int(num_cls)
        try:
            cluster_news.retrieve(q, "2021-03-17", "2021-03-18")
        except:
            flash("Invalid Date")
            return redirect(url_for("home"))
        dataframe = cluster_news.retrieve(q, start_date, end_date)
        data = cluster_news.organize(dataframe)
        doc_term_mat, df, w2id, bibs = vl.read_df(data, dataset, stopwords)

    elif dataset == "Experts":
        import es_search_experts

        data, num_docs, error = es_search_experts.retrieve(q)

        if error == None:
            import experts_stopwords

            exp_stopwords = experts_stopwords.stopwords
            exp_stopwords.update(stopwords)
            num_cls = get_num_cls_for_experts(num_docs)
            if entity == "experts":
                (
                    orig_doc_term_mat,
                    orig_df,
                    df,
                    dfr,
                    keywords,
                    cluster_centers,
                    cluster_desc,
                    coordinates,
                    id2members,
                    bibs,
                    org_ids,
                ) = vl.process_experts_by_name(data, dataset, num_cls, exp_stopwords)
            else:  # entity == 'topics'
                (
                    orig_doc_term_mat,
                    orig_df,
                    df,
                    dfr,
                    keywords,
                    cluster_centers,
                    cluster_desc,
                    coordinates,
                    id2members,
                    bibs,
                    org_ids,
                ) = vl.process_experts_by_topic(data, dataset, num_cls, exp_stopwords)

            # doc_term_mat, df, w2id, bibs = vl.read_df_experts(
            #         data, dataset, exp_stopwords)
        else:
            # flash(error)
            return redirect(url_for("home"))

    elif dataset == "DigiSquare":
        import es_search_DigiSquare

        data, num_docs, error = es_search_DigiSquare.retrieve(q)

        num_cls = int(num_cls)

        if error == None:
        ##########################################
            # if entity == 'genes':
            #    doc_term_mat, df, w2id, bibs = vl.read_df_genes(
            #        session['gene_set'], data, dataset, stopwords)
            #    # print('-----------df------------')
            #    # print(df)
            # elif entity == 'authors':
            #    # doc_term_mat, df, w2id, bibs = vl.read_df_authors(
            #    #     data, dataset, stopwords)
            #    orig_doc_term_mat, orig_df, df, dfr, keywords, cluster_centers, cluster_desc, coordinates, id2members, bibs, org_ids, author_names, author_docs = vl.process_authors_4(data, dataset, stopwords)
            # else:
        ##########################################
            doc_term_mat, df, w2id, bibs = vl.read_df_digi(data, dataset, stopwords)
        else:
            # flash(error)
            return redirect(url_for("home"))

    elif dataset == "PubMedAPI":
        print("PubMedAPI in routes.py")
        import cluster_pymedAPI
        import es_search

        # print("start_date")
        # print(start_date)
        num_cls = int(num_cls)
        # try:
        # data, error = es_search.retrieve(q, entity)
        data, error = cluster_pymedAPI.retrieve(q)
        # data, error = get_dc_data(q)
        # http://localhost:5000/cluster?dataset_opt=PubMedAPI&query=Javed Mostafa
        # http://localhost:5000/cluster?dataset_opt=PLOS&query=digital%20health
        # http://localhost:5000/cluster?dataset_opt=NewsAPI&query=digital%20health
        print(data, error)
        print(type(data), data.columns.tolist())
        # except:
        #     flash("The National Library of Medicine Entrez API is experiencing technical issues. We're sorry for this inconvenience.")
        #     return redirect(url_for('home'))

        if error == None:
            if entity == "genes":
                doc_term_mat, df, w2id, bibs = vl.read_df_genes(
                    session["gene_set"], data, dataset, stopwords
                )
                # print('-----------df------------')
                # print(df)
            elif entity == "authors":
                # doc_term_mat, df, w2id, bibs = vl.read_df_authors(
                #     data, dataset, stopwords)
                (
                    orig_doc_term_mat,
                    orig_df,
                    df,
                    dfr,
                    keywords,
                    cluster_centers,
                    cluster_desc,
                    coordinates,
                    id2members,
                    bibs,
                    org_ids,
                    author_names,
                    author_docs,
                ) = vl.process_authors_4(data, dataset, stopwords)
            else:
                doc_term_mat, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
        else:
            # flash(error)
            return redirect(url_for("home"))

    elif (
        dataset == "GoogleAPI"
        or dataset == "GoogleAPI2"
        or dataset == "GoogleAPI3"
        or dataset == "GoogleAPI4"
        or dataset == "GoogleAPI5"
    ):
        doc_term_mat, df, w2id, bibs = google_cluster(dataset, num_cls)

    # solr indexed static dataset
    elif (
        dataset == "NYTIMES"
        or dataset == "PLOS"
        or dataset == "DIABETES"
        or dataset == "COVID19"
    ):
        # Setup a Solr instance. The timeout is optional.
        if dataset == "NYTIMES":
            SOLR = "http://localhost:8983/solr/nytimes"
        elif dataset == "PLOS":
            # SOLR = 'http://localhost:8080/solr/plos'
            if False:  # os.uname().sysname == 'Linux':
                SOLR = "http://localhost:8983/solr/plos2"
            else:
                SOLR = "https://api.plos.org/search?"  # plos server
                # SOLR = 'http://3.18.126.137:8983/solr/nytimes'   # aws instance
                # SOLR = 'http://localhost:8080/solr/pmc'   # port forwarding
                # SOLR = 'http://vzlib:G8RWB2sF@localhost:8983/solr/plos' # password
        elif dataset == "DIABETES":
            SOLR = "http://localhost:8983/solr/diabetes"
        elif dataset == "COVID19":
            SOLR = "http://localhost:8983/solr/covid19"

        solr = pysolr.Solr(SOLR, timeout=10)

        # print(q)
        # print(field)
        # print(next)
        if q == "":
            q = "*"

        if field == "Title":
            query = "title:" + q
        elif field == "Abstract":
            query = "abstract:" + q
        elif field == "Body":
            query = "body:" + q
        else:
            query = "everything:" + q

        num_cls = int(num_cls)

        if SOLR == "https://api.plos.org/search?":
            # for connecting to PLOS server
            solr_tuples = [
                ("q", query),
                ("fl", "title,pmid,abstract,author," "journal_name,publication_date"),
                ("rows", "500"),
                ("wt", "json"),
            ]
            # ("sort", "publication_date desc")]
            encoded_solr_tuples = urlencode(solr_tuples)
            connection = urlopen(SOLR + encoded_solr_tuples)
            # print("1Retrieving data from PLOS server...")
            print(encoded_solr_tuples)  # newly added
            response = simplejson.load(connection)
            doc_term_mat, df, w2id, bibs = vl.read_json(response, dataset)
            # newly added
            total = len(doc_term_mat)
            print(total)
            if total == 0:
                flash("No Result Has been Found.")
                return redirect(url_for("home"))

        elif SOLR == "http://localhost:8983/solr/plos2":
            doc_term_mat = solr.search(
                query,
                **{
                    "rows": "500",  # number of articles to retrieve
                    "fl": "title,pmid,abstract,author,journal_name," "publication_date",
                    "sort": "pmid asc",
                    # 'sort': 'publication_date desc'
                },
            )

            total = len(doc_term_mat)
            print(total)
            if total == 0:
                flash("No Results Have been Found.")
                return redirect(url_for("home"))
            # for d in doc_term_mat:
            #   print("The title is '{0}'.".format(d['title']))

            # Read documents
            print("2Retrieving data from Solr server...")
            doc_term_mat, df, w2id, bibs = vl.read_pysolr(
                doc_term_mat, dataset, stopwords
            )

        elif SOLR == "http://localhost:8983/solr/covid19":
            doc_term_mat = solr.search(
                query,
                **{
                    "rows": "1000",  # number of articles to retrieve
                    "fl": "title,pmid,author,body",
                    # 'sort': 'publication_date desc'
                },
            )

            total = len(doc_term_mat)
            print(total)
            if total == 0:
                flash("No Results Have been Found.")
                return redirect(url_for("home"))
            # for d in doc_term_mat:
            #   print("The title is '{0}'.".format(d['title']))

            # Read documents
            print("2Retrieving data from Solr server...")
            doc_term_mat, df, w2id, bibs = vl.read_pysolr(
                doc_term_mat, dataset, stopwords
            )

        elif SOLR == "http://localhost:8983/solr/nytimes":
            doc_term_mat = solr.search(
                query,
                **{
                    "rows": "500",  # number of articles to retrieve
                    "fl": "title,body,html",
                },
            )

            total = len(doc_term_mat)
            print(total)
            if total == 0:
                flash("No Results Have been Found.")
                return redirect(url_for("home"))
            # for d in doc_term_mat:
            #   print("The title is '{0}'.".format(d['title']))

            # Read documents
            print("3Retrieving data from Solr server...")
            doc_term_mat, df, w2id, bibs = vl.read_pysolr(
                doc_term_mat, dataset, stopwords
            )

        elif SOLR == "http://localhost:8983/solr/diabetes":
            doc_term_mat = solr.search(
                query,
                **{
                    "rows": "500",  # number of articles to retrieve
                    "fl": "title,pmid,abstract,author," "publication_date",
                    "sort": "pmid asc",
                },
            )

            total = len(doc_term_mat)
            print(total)
            if total == 0:
                flash("No Results Have been Found.")
                return redirect(url_for("home"))
            # for d in doc_term_mat:
            #   print("The title is '{0}'.".format(d['title']))

            # Read documents
            print("2Retrieving data from Solr server...")
            doc_term_mat, df, w2id, bibs = vl.read_pysolr(
                doc_term_mat, dataset, stopwords
            )

        print("Finished reading %d documents" % len(doc_term_mat))

    elif dataset == "PostgreSQL":
        print("PostgreSQL in routes.py")
        import cluster_postgreSQL
        from cluster_postgreSQL import retrieve_from_postgresql, execute_query, search_database_by_keyword, optimal_number_of_clusters
        num_cls = int(num_cls)
        data, error = retrieve_from_postgresql(q)
        print(data, error)
        print(type(data), data.columns.tolist())
        if error == None:
            if entity == "genes":
                doc_term_mat, df, w2id, bibs = vl.read_df_genes(
                    session["gene_set"], data, dataset, stopwords
                )
            elif entity == "authors":
                (
                    orig_doc_term_mat,
                    orig_df,
                    df,
                    dfr,
                    keywords,
                    cluster_centers,
                    cluster_desc,
                    coordinates,
                    id2members,
                    bibs,
                    org_ids,
                    author_names,
                    author_docs,
                ) = vl.process_authors_4(data, dataset, stopwords)
            else:
                doc_term_mat, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
             
     
    elif dataset == "rss_feed":
        print("rss_feed in routes.py")
        import cluster_postgreSQL
        from cluster_postgreSQL import retrieve_from_postgresql, execute_query, search_database_by_keyword, optimal_number_of_clusters
        num_cls = int(num_cls)
        # data, error = retrieve_from_postgresql(q)
        # data, error = retrieve_data(q)
        data, error = search_database_by_keyword(q)
        print(data, error)
        print(type(data), data.columns.tolist())
        if error == None:
            if entity == "genes":
                doc_term_mat, df, w2id, bibs = vl.read_df_genes(
                    session["gene_set"], data, dataset, stopwords
                )
            elif entity == "authors":
                (
                    orig_doc_term_mat,
                    orig_df,
                    df,
                    dfr,
                    keywords,
                    cluster_centers,
                    cluster_desc,
                    coordinates,
                    id2members,
                    bibs,
                    org_ids,
                    author_names,
                    author_docs,
                ) = vl.process_authors_4(data, dataset, stopwords)
            else:
                doc_term_mat, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
   
    # uploaded dataset
    else:
        # num_cls = 10
        uploaded_file = dataset
        # path = UPLOAD_FOLDER  # needs to be changed
        # print(path)
        doc_term_mat, df, w2id, bibs = vl.read_plaintext(
            path, uploaded_file, q, dataset, stopwords
        )

    # print("doc_term_mat")
    # print(doc_term_mat)
    # print(df)

    if dataset != "Experts" and len(bibs) < 10:
        #       flash('No clusters found. Broaden your search.')
        flash(
            "Please upload larger files or search for more relevant keywords to get enough results."
        )
        #        return redirect(url_for('home'))
        return redirect(request.referrer)

    if entity != "authors" and dataset != "Experts":
        # Remove terms whose df is lower than mindf
        if MINDF > 0:
            inf = []
            for w in df:
                if df[w] <= MINDF:
                    inf.append(w)
            for w in inf:
                del df[w]

        # Save org data
        orig_doc_term_mat = doc_term_mat
        orig_df = df

        # Compute tfidf and find key terms
        # print("Computing TFIDF and finding key terms...")
        if dataset == "NYTIMES" or dataset == "PLOS" or dataset == "DIABETES":
            doc_term_mat, dfr = vl.compute_tfidf(doc_term_mat, df, rank=10)
        else:
            doc_term_mat, dfr = vl.compute_tfidf(doc_term_mat, df, rank=30)

        # Sort and output results (discovered keywords)
        keywords = vl.output_keywords(len(doc_term_mat), dfr, df, p_docs=1.0)
        # print('Keywords...')
        # print(keywords)

        # Create new matrix with the keywords
        doc_term_mat, org_ids = vl.update(doc_term_mat, keywords)

        # Convert to sparse matrix
        doc_term_mat = vl.convert_sparse(doc_term_mat, keywords)

        # Clustering
        # print()
        # print("Clustering...")
        # Determine the best number of clusters if not provided
        if num_cls is None:
            num_cls = optimal_number_of_clusters(doc_term_mat, max_clusters=10)
        else:
            num_cls = int(num_cls)

        print(f"Optimal number of clusters: {num_cls}")
        

        # n_components: number of dimensions for LSA
        # k: number of clusters
        # n_desc: number of keywords (desc) for each cluster

        if dataset == "PLOS":
            id2members, cluster_centers, cluster_desc, coordinates, error = vl.kmeans(
                doc_term_mat, keywords, org_ids, n_components=50, k=num_cls, n_desc=25
            )
        else:
            id2members, cluster_centers, cluster_desc, coordinates, error = vl.kmeans(
                doc_term_mat, keywords, org_ids, n_components=20, k=num_cls, n_desc=15
            )

        if error != None:  # needs to be changed
            print(error)
            flash(
                "Sorry, the entity you've chosen does not generate meaningful clusters for current query. Please try other entities or queries. "
            )
            return redirect(request.referrer)

    # get cluster colors
    """
    # center is (.5,0)
    cc = np.array(cluster_centers)-[.5,0]
    hue = ((np.arctan2(cc[:,1],cc[:,0])/np.pi+1)*180*2).\
        astype(int).tolist()
    satr = (np.linalg.norm(cc, axis=1)/math.sqrt(5)*2).tolist()
    """
    # center is (.5,.5)
    cc = np.array(cluster_centers) - 0.5
    hue = ((np.arctan2(cc[:, 1], cc[:, 0]) / np.pi + 1) * 180).astype(int).tolist()
    satr = (np.linalg.norm(cc, axis=1) / math.sqrt(2) * 2).tolist()
    val = [0.8] * num_cls

    # Create data to pass to result.html
    df_new = dict()
    dfr_new = dict()
    for w in keywords:
        df_new[w] = df[w]
        dfr_new[w] = dfr[w]

    # create adjacency matrix that will be used for network edges
    centroid_matrix = pd.DataFrame.from_records(cluster_centers)
    centroid_distances = pd.DataFrame(
        squareform(pdist(centroid_matrix, metric="euclidean")),
        columns=list(id2members.keys()),
        index=list(id2members.keys()),
    ).to_dict()

    id2freq = {x: len(id2members[x]) for x in id2members}

    # create network data structure
    # print('\nEUCLIDEAN DISTANCES\n----------')
    edges = []
    for i in id2members:
        for k, v in centroid_distances[i].items():
            # print(i, ' to ', k, ' = ', v)
            if i != k:
                if v > 0.75:
                    v = "Distant"
                    edge = {
                        "clusterID": i,
                        "source": cluster_centers[i],
                        "target": cluster_centers[k],
                        "distance": v,
                    }
                    edges.append(edge)
                elif v <= 0.75 and v > 0.50:
                    v = "Similar"
                    edge = {
                        "clusterID": i,
                        "source": cluster_centers[i],
                        "target": cluster_centers[k],
                        "distance": v,
                    }
                    edges.append(edge)
                elif v <= 0.50:
                    v = "Very Similar"
                    edge = {
                        "clusterID": i,
                        "source": cluster_centers[i],
                        "target": cluster_centers[k],
                        "distance": v,
                    }
                    edges.append(edge)

                # elif v >= 0.50:
                #     v = "Not Similar"
                # edge = {"clusterID":i,"source":cluster_centers[i],"target":cluster_centers[k],"distance":v}
                # edges.append(edge)
    # print('\nBIBLIOGRAPHY\n----------')
    sources = []
    id2meshTerms = {}
    if dataset == "PubMedAPI":
        from collections import Counter

        for cluster_id in sorted(id2members.keys()):
            concepts = cluster_desc[cluster_id]
            references = []
            for bib_id in id2members[cluster_id]:
                references.append(bibs[bib_id])
            for paper in references:
                if type(paper["author"]) != str and None in paper["author"]:
                    paper["author"] = [i for i in paper["author"] if i]
            bibliography = {"concepts": concepts, "references": references}
            meshTerms = []
            for ref in references:
                meshHeadingList = ref.get("meshHeadings")  # ref["meshHeadings"]
                for mesh in meshHeadingList:
                    meshTerms.append(mesh["descriptor"])
            # print('concept: {}'.format(concepts))
            # print('meshTerms: {}'.format(Counter(meshTerms)))
            # print('\n')
            id2meshTerms[cluster_id] = str(dict(Counter(meshTerms).most_common(20)))
            # vl.genes_coverage(concepts, references)
            sources.append(bibliography)
        # print(sources)

    if dataset == "PostgreSQL":
        from collections import Counter

        for cluster_id in sorted(id2members.keys()):
            concepts = cluster_desc[cluster_id]
            references = []
            print(f'--------{cluster_id}---------')
            for bib_id in id2members[cluster_id]:
                references.append(bibs[bib_id])
            print(references)
            print('-----------------')
            for paper in references:
                if type(paper["author"]) != str and None in paper["author"]:
                    paper["author"] = [i for i in paper["author"] if i]
                    print(paper["author"])
            # Add 'meshHeadings': [] to each reference in references
            for ref in references:
                ref['meshHeadings'] = []
            bibliography = {"concepts": concepts, "references": references}
            meshTerms = []
            for ref in references:
                meshHeadingList = ref.get("meshHeadings")  # ref["meshHeadings"]
                for mesh in meshHeadingList:
                    meshTerms.append(mesh["descriptor"])
            id2meshTerms[cluster_id] = str(dict(Counter(meshTerms).most_common(20)))
            sources.append(bibliography)
            
    if dataset == "rss_feed":
        from collections import Counter

        for cluster_id in sorted(id2members.keys()):
            concepts = cluster_desc[cluster_id]
            references = []
            print(f'--------{cluster_id}---------')
            for bib_id in id2members[cluster_id]:
                references.append(bibs[bib_id])
            # print(references)
            print('-----------------')
            # for paper in references:
            #     if type(paper["author"]) != str and None in paper["author"]:
            #         paper["author"] = [i for i in paper["author"] if i]
            #         print(paper["author"])
            # Add 'meshHeadings': [] to each reference in references
            for ref in references:
                ref['meshHeadings'] = []
            bibliography = {"concepts": concepts, "references": references}
            meshTerms = []
            for ref in references:
                meshHeadingList = ref.get("meshHeadings")  # ref["meshHeadings"]
                for mesh in meshHeadingList:
                    meshTerms.append(mesh["descriptor"])
            id2meshTerms[cluster_id] = str(dict(Counter(meshTerms).most_common(20)))
            sources.append(bibliography)

    """
    cnt = np.unique(membership, return_counts=True)
    keys = [x for x in cnt[0]]
    values = [int(x) for x in cnt[1]]
    id2freq = dict(zip(keys, values))
    """

    """
    # Prepare bib data if num of documents is small
    if len(org_ids) > 150:
        bibs = []
    """
    import datetime
    import cluster_postgreSQL
    from cluster_postgreSQL import get_date_summary
    ear_date, lat_date, last_up = get_date_summary()
    # session['doc_term_mat'] = doc_term_mat  # term-doc matrix
    session["docs_org"] = orig_doc_term_mat  # doc-term raw freq matrix
    session["df_org"] = orig_df
    session["num_cls"] = num_cls
    session["id2members_0"] = id2members
    session["cluster_desc_0"] = cluster_desc
    session["xy_0"] = cluster_centers
    session["state"] = state
    session["hue_0"] = hue
    session["satr_0"] = satr
    session["org_ids_0"] = org_ids
    session["bibs_0"] = bibs
    session["dataset"] = dataset
    session["edges_0"] = edges
    session["sources"] = sources
    session["id2meshTerms"] = id2meshTerms
    session["ear_date"] = ear_date
    session["lat_date"] = lat_date
    session["last_up"] = last_up

    session["entity"] = entity
    if entity == "authors":
        session["author_names"] = author_names
        session["author_docs"] = author_docs
        
    # Convert NumPy arrays to lists
    def convert_ndarray_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_ndarray_to_list(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
        else:
            return obj

    # Ensure orig_doc_term_mat is a list
    orig_doc_term_mat_list = convert_ndarray_to_list(orig_doc_term_mat)
    # Ensure orig_df is a list or dict (depending on its original type)
    orig_df_list = convert_ndarray_to_list(orig_df)
    # Ensure cluster_centers is a list
    cluster_centers_list = convert_ndarray_to_list(cluster_centers)
    # Ensure coordinates is a list
    coordinates_list = convert_ndarray_to_list(coordinates)

    
    
    # Generate summaries for each cluster
    cluster_summary = {}
    cluster_assigned = {}
    cluster_title = {}
    for cluster_id in sorted(id2members.keys()):
        publications = id2members[cluster_id]
        pub_texts = "\n".join([f"@article{{{bib_id}, title={bibs[bib_id]['title']}}}" for bib_id in publications])
        # user_content = f"Clustered Publications:\n\n{pub_texts}\n\nAssigned Cluster: '{cluster_desc[cluster_id]}'"
        user_content = f"Clustered Publications:\n\n{pub_texts}'"
        # print(f"Cluster {cluster_id} \n\n User Content: {user_content}")
        print(f"Cluster {cluster_id} \n\n User Content: {user_content}\n\n")
        summary = generate_summary(user_content)
        # Split the string into parts based on the numbering
        print("Preprocessed Summary: ", summary, "\n\n")
        parts = summary.split("2. ")

        summary_title = parts[0].replace("1. ", "").strip()
        summary_val = parts[1].strip()
        print(f"Cluster {cluster_id}, Assigned Cluster: '{cluster_desc[cluster_id]}'\n\nSummary: {summary_val}\n\nTitle: {summary_title}\n\n")
        if 'N/A' in summary_val or 'N/A' in summary_title:
            summary_val = "No summary available due to insufficient paper content"
            summary_title = "No title available"
        cluster_summary[cluster_id] = summary_val
        cluster_assigned[cluster_id] = cluster_desc[cluster_id]
        # If summary title is a sentence, i want it to be a list of words
        summary_title = summary_title.split()
        cluster_title[cluster_id] = summary_title
    
    cluster_summary_list = [[value] for key, value in cluster_summary.items()]
    cluster_title_list = [[value] for key, value in cluster_title.items()]
   
    session["cluster_desc_0"] = cluster_title_list
    def unpack_list(lst):
        return [item for sublist in lst for item in sublist]
    
    sessionData = {
        "docs_org": convert_ndarray_to_list(orig_doc_term_mat),  # doc-term raw freq matrix
        "df_org": convert_ndarray_to_list(orig_df),
        "num_cls": num_cls,
        "id2members_0": convert_ndarray_to_list(id2members),
        # "cluster_desc_0": convert_ndarray_to_list(cluster_desc),
        "cluster_desc_0": unpack_list(convert_ndarray_to_list(cluster_title_list)),
        "cluster_summary_0": convert_ndarray_to_list(cluster_summary_list),
        "xy_0": convert_ndarray_to_list(cluster_centers),
        "state": state,
        "hue_0": convert_ndarray_to_list(hue),
        "satr_0": convert_ndarray_to_list(satr),
        "org_ids_0": convert_ndarray_to_list(org_ids),
        "dataset": dataset,
        "edges_0": convert_ndarray_to_list(edges),
        "sources": convert_ndarray_to_list(sources),
        "ear_date": ear_date,
        "lat_date": lat_date,
        "last_up": last_up
    }

    # session['df'] = df_new
    # session['dfr'] = dfr_new
    # session['keywords'] = keywords


    #    username = 'username'

    # set username if onyen is detected, make a new upload directory for new user
    #    if 'eppn' in env:
    #        username = env['eppn'].split('@')[0]

    # return render_template("result.html",
    receive_time = time.clock()
    response_time = str(round(receive_time - send_time, 3))
    
    # Print the cluster_title_list
    print("cluster_title_list of sessionData: ", cluster_title_list)
    print("cluster_title of render_template: ", cluster_title)
    print("cluster_title_list in converted form: ", convert_ndarray_to_list(cluster_title_list))
    

    if entity == "authors" or entity == "experts" or entity == "topics":
        # to tell the difference between keyword search and author search so that sessionStorage won't keep old data when switching between them
        q = q + "_" + entity
    print("state in cluster: ", state)
    print("exit cluster")
    # return render_template(
    #     "results.html",
    #     sessionData=sessionData,
    #     login=login,
    #     loginType=loginType,
    #     username=username,
    #     query=raw_query,
    #     decoratedQuery=q,
    #     error=error,
    #     keywords=keywords,
    #     df=df_new,
    #     dfr=dfr_new,
    #     # cluster_desc=cluster_desc,
    #     cluster_desc=cluster_title,
    #     cluster_summary = cluster_summary,
    #     xy=cluster_centers_list,
    #     edges=edges,
    #     id2freq=id2freq,
    #     xy_inter=coordinates_list,
    #     hue=hue,
    #     satr=satr,
    #     val=val,
    #     id2members=id2members,
    #     sources=sources,
    #     dataset=dataset,
    #     response_time=response_time,
    #     path=path,
    #     num_cls=num_cls,
    #     ear_date=ear_date,
    #     lat_date=lat_date,
    #     last_up=last_up,
    # )
    session["sessionData"] = sessionData
    session["results_data"] = {
                                "login": login, "loginType": loginType, "username": username, "query": raw_query, "decoratedQuery": q, "error": error, 
                                "keywords": keywords, "df": df_new, "dfr": dfr_new, "cluster_desc": cluster_title, "cluster_summary": cluster_summary, 
                                "xy": cluster_centers_list, "edges": edges, "id2freq": id2freq, "xy_inter": coordinates_list, "hue": hue, "satr": satr, 
                                "val": val, "id2members": id2members, "sources": sources, "dataset": dataset, "response_time": response_time, "path": path, 
                                "num_cls": num_cls, "ear_date": ear_date, "lat_date": lat_date, "last_up": last_up
                            }            
    return redirect(url_for("calibration", 
                            query=raw_query,
                            num_cls=num_cls,
    ))
    
    


# process user modeling data
@app.route("/user_modeling", methods=["POST", "GET"])
def user_modeling():
    print("user_modeling")
    if request.method == "POST":
        # parse interaction data
        click_data = request.get_json()
        # serialize click data to pretty JSON
        json_object = json.dumps(click_data, indent=4)

        # get the user information
        env = request.environ
        if "eppn" in env:
            username = env["eppn"].split("@")[0]
            click_data["username"] = username
        elif current_user.is_authenticated:
            username = current_user.username
            click_data["username"] = username
        else:
            print("No user logged in for user modeling")

            # for local testing
        #        username = 'msortiz'
        #        click_data['username'] = username

        # write this data to the users index
        es = Elasticsearch()  # make sure elasticsearch service is running
        res = es.index(index="users", body=click_data)
        # print(res['result'])
        # print('\n\n')

        # test search operations
        #        res = es.search(index="users", body={"query": {"match_all": {}}})
        #        print(res)
        #        print('\n\n')

    return "", 200


# for local
@app.route("/csv/<filename>")
def csv_file(filename):
    print("csv_file")
    return send_from_directory(CSV_DIR, filename)


# for server
@app.route("/..")
def csv_folder():
    print("csv_folder")
    return send_from_directory(CSV_DIR)


# for server
@app.route("/help")
def help():
    print("help")
    return render_template("help.html")

# --- TOBII GAZE STREAMING + CALIBRATION ---



# Real-time streaming logic
def stream_gaze_data():
    # Get the Tobii Eye Tracker device
    eye_tracker = tr.find_all_eyetrackers()[0]
    def gaze_callback(gaze_data):
        if gaze_data.get("left_gaze_point_on_display_area"):
            x, y = gaze_data["left_gaze_point_on_display_area"]
            if 0 <= x <= 1 and 0 <= y <= 1:
                socketio.emit("gaze_data", {"x": x, "y": y})
    
    eye_tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_callback, as_dictionary=True)

@socketio.on("start_gaze")
def start_gaze():
    threading.Thread(target=stream_gaze_data).start()
    emit("gaze_status", {"status": "started"})

@app.route("/start_calibration", methods=["POST"])
def start_calibration():
    try:
        # Get the Tobii Eye Tracker device
        eye_tracker = tr.find_all_eyetrackers()[0]
        calibration = tr.ScreenBasedCalibration(eye_tracker)
        calibration.enter_calibration_mode()

        points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
        ]

        for point in points:
            calibration.collect_data(point)
            time.sleep(0.3)

        result = calibration.compute_and_apply()
        calibration.leave_calibration_mode()

        return jsonify({"status": "success", "message": f"Calibration complete. Quality: {result.status}"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})