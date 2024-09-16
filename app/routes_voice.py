# @app.route('/pattie_voice')
# def home_voice():
#   path = UPLOAD_FOLDER  # needs to be changed
#
#   uploaded_files = []
#   # r=root, d=directories, f = files
#   for r, d, f in os.walk(path):
#       for file in f:
#           if '.txt' in file or '.csv' in file:
#               uploaded_files.append(file)
#   uploaded_files.append("Upload...")
#   field = 'All fields'
#   today = datetime.today().strftime('%Y-%m-%d')
#   yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
#   return render_template('index_voice.html', field=field, uploaded_files=uploaded_files,
#                          start_date=yesterday, end_date=today)

# @app.route('/cluster_voice', methods=['POST'])
# def cluster_voice():
#
#   response_time = '0'
#   state = 0
#   send_time = time.clock()
#   error = None
#   dataset = request.form['dataset_opt']
#   field = request.form['field']
#   q = request.form['query']
#   # num_cls = request.form['num_cls']
#   # if num_cls == 'Optional: # of clusters':
#   num_cls = 10
#
#   if dataset == 'Placeholder':
#       return redirect(url_for('home'))
#
#   # dynamic dataset
#   elif dataset == "NewsAPI":
#       import cluster_news
#       start_date = request.form['start_date']
#       end_date = request.form['end_date']
#       num_cls = int(num_cls)
#       try:
#           cluster_news.retrieve(q, start_date, end_date)
#       except:
#           flash("Invalid Date")
#           return redirect(url_for('home'))
#       dataframe = cluster_news.retrieve(q, start_date, end_date)
#       data = cluster_news.organize(dataframe)
#       docs, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
#
#   elif dataset == "PubMedAPI":
#       import cluster_pymedAPI
#       # print("start_date")
#       # print(start_date)
#       num_cls = int(num_cls)
#       data, error = cluster_pymedAPI.retrieve(q)
#       if error == None:
#           docs, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
#       else:
#           # flask(error)
#           return redirect(url_for('home'))
#
#   elif dataset == "GoogleAPI":
#       import googleAPI
#       # print("start_date")
#       # print(start_date)
#       num_cls = int(num_cls)
#       # if no query is entered exit the function and redirect url to home
#       if q == '':
#           return redirect(url_for('home'))
#       data, error = googleAPI.retrieve(q)
#       if data.empty:
#           return redirect(url_for('home'))
#       if error == None:
#           docs, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
#       else:
#           # #flask(error)
#           return redirect(url_for('home'))
#
#   elif dataset == "GoogleAPI2":
#       import googleAPI2
#       # print("start_date")
#       # print(start_date)
#       num_cls = int(num_cls)
#       # if no query is entered exit the function and redirect url to home
#       if q == '':
#           return redirect(url_for('home'))
#       data, error = googleAPI2.retrieve(q)
#       if data.empty:
#           return redirect(url_for('home'))
#       if error == None:
#           docs, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
#       else:
#           # #flask(error)
#           return redirect(url_for('home'))
#
#   elif dataset == "GoogleAPI3":
#       import googleAPI3
#       # print("start_date")
#       # print(start_date)
#       num_cls = int(num_cls)
#       # if no query is entered exit the function and redirect url to home
#       if q == '':
#           return redirect(url_for('home'))
#       data, error = googleAPI3.retrieve(q)
#       if data.empty:
#           return redirect(url_for('home'))
#       if error == None:
#           docs, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
#       else:
#           # #flask(error)
#           return redirect(url_for('home'))
#
#   elif dataset == "GoogleAPI4":
#       import googleAPI4
#       # print("start_date")
#       # print(start_date)
#       num_cls = int(num_cls)
#       # if no query is entered exit the function and redirect url to home
#       if q == '':
#           return redirect(url_for('home'))
#       data, error = googleAPI4.retrieve(q)
#       if data.empty:
#           return redirect(url_for('home'))
#       if error == None:
#           docs, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
#       else:
#           # #flask(error)
#           return redirect(url_for('home'))
#
#   elif dataset == "GoogleAPI5":
#       import googleAPI5
#       # print("start_date")
#       # print(start_date)
#       num_cls = int(num_cls)
#       # if no query is entered exit the function and redirect url to home
#       if q == '':
#           return redirect(url_for('home'))
#       data, error = googleAPI5.retrieve(q)
#       if data.empty:
#           return redirect(url_for('home'))
#       if error == None:
#           docs, df, w2id, bibs = vl.read_df(data, dataset, stopwords)
#       else:
#           # #flask(error)
#           return redirect(url_for('home'))
#
#   # solr indexed static dataset
#   elif dataset == "NYTIMES" or dataset == "PLOS" or dataset == "DIABETES":
#       # Setup a Solr instance. The timeout is optional.
#       if dataset == "NYTIMES":
#           SOLR = 'http://localhost:8983/solr/nytimes'
#       elif dataset == "PLOS":
#           # SOLR = 'http://localhost:8080/solr/plos'
#           if os.uname().sysname == 'Linux':
#               SOLR = 'http://localhost:8983/solr/plos2'
#           else:
#               SOLR = 'http://api.plos.org/search?'   # plos server
#               # SOLR = 'http://3.18.126.137:8983/solr/nytimes'   # aws instance
#               # SOLR = 'http://localhost:8080/solr/pmc'   # port forwarding
#               # SOLR = 'http://vzlib:G8RWB2sF@localhost:8983/solr/plos' # password
#       elif dataset == "DIABETES":
#           SOLR = 'http://localhost:8983/solr/diabetes'
#
#       solr = pysolr.Solr(SOLR, timeout=10)
#
#       # print(q)
#       # print(field)
#       # print(next)
#       if q == "":
#           q = '*'
#
#       if field == 'Title':
#           query = 'title:' + q
#       elif field == 'Abstract':
#           query = 'abstract:' + q
#       elif field == 'Body':
#           query = 'body:' + q
#       else:
#           query = 'everything:' + q
#
#       num_cls = int(num_cls)
#
#       if SOLR == 'http://api.plos.org/search?':
#           # for connecting to PLOS server
#           solr_tuples = [("q", query),
#                          ("fl", "title,pmid,abstract,author,"
#                           "journal_name,publication_date"),
#                          ("rows", "500"), ("wt", "json")]
#           # ("sort", "publication_date desc")]
#           encoded_solr_tuples = urlencode(solr_tuples)
#           connection = urlopen(SOLR + encoded_solr_tuples)
#           print("1Retrieving data from PLOS server...")
#           print(encoded_solr_tuples)  # newly added
#           response = simplejson.load(connection)
#           docs, df, w2id, bibs = \
#               vl.read_json(response, dataset)
#           # newly added
#           total = len(docs)
#           print(total)
#           if total == 0:
#               flash("No Result Has been Found.")
#               return redirect(url_for('home'))
#
#       elif SOLR == 'http://localhost:8983/solr/plos2':
#           docs = solr.search(query, **{
#               'rows': '500',  # number of articles to retrieve
#               'fl': 'title,pmid,abstract,author,journal_name,'
#               'publication_date',
#               'sort': 'pmid asc'
#               # 'sort': 'publication_date desc'
#           })
#
#           total = len(docs)
#           print(total)
#           if total == 0:
#               flash("No Results Have been Found.")
#               return redirect(url_for('home'))
#           # for d in docs:
#           #   print("The title is '{0}'.".format(d['title']))
#
#           # Read documents
#           print("2Retrieving data from Solr server...")
#           docs, df, w2id, bibs = vl.read_pysolr(docs, dataset, stopwords)
#
#       elif SOLR == 'http://localhost:8983/solr/nytimes':
#           docs = solr.search(query, **{
#               'rows': '500',  # number of articles to retrieve
#               'fl': 'title,body,html'
#           })
#
#           total = len(docs)
#           print(total)
#           if total == 0:
#               flash("No Results Have been Found.")
#               return redirect(url_for('home'))
#           # for d in docs:
#           #   print("The title is '{0}'.".format(d['title']))
#
#           # Read documents
#           print("3Retrieving data from Solr server...")
#           docs, df, w2id, bibs = vl.read_pysolr(docs, dataset, stopwords)
#
#       elif SOLR == 'http://localhost:8983/solr/diabetes':
#           docs = solr.search(query, **{
#               'rows': '500',  # number of articles to retrieve
#               'fl': 'title,pmid,abstract,author,'
#               'publication_date',
#               'sort': 'pmid asc'
#           })
#
#           total = len(docs)
#           print(total)
#           if total == 0:
#               flash("No Results Have been Found.")
#               return redirect(url_for('home'))
#           # for d in docs:
#           #   print("The title is '{0}'.".format(d['title']))
#
#           # Read documents
#           print("2Retrieving data from Solr server...")
#           docs, df, w2id, bibs = vl.read_pysolr(docs, dataset, stopwords)
#
#       print("Finished reading %d documents" % len(docs))
#
#   # uploaded dataset
#   else:
#       num_cls = int(num_cls)
#       uploaded_file = dataset
#       path = UPLOAD_FOLDER  # needs to be changed
#       # path_to_file = path + uploaded_file
#       docs, df, w2id, bibs = vl.read_plaintext(
#           path, uploaded_file, q, dataset, stopwords)
#
#   # print("docs")
#   # print(docs)
#   print(df)
#   # Remove terms whose df is lower than mindf
#   if MINDF > 0:
#       inf = []
#       for w in df:
#           if df[w] <= MINDF:
#               inf.append(w)
#       for w in inf:
#           del df[w]
#
#   print("docs")
#   # print(docs)
#
#   # Save org data
#   session['docs_org'] = docs  # term-doc raw freq matrix
#   session['df_org'] = df
#
#   # Compute tfidf and find key terms
#   print("Computing TFIDF and finding key terms...")
#   if dataset == "NYTIMES" or dataset == "PLOS" or dataset == "DIABETES":
#       docs, dfr = vl.compute_tfidf(docs, df, rank=10)
#   else:
#       docs, dfr = vl.compute_tfidf(docs, df, rank=30)
#
#   # Sort and output results (discovered keywords)
#   keywords = vl.output_keywords(len(docs), dfr,
#                                 df, p_docs=1.0)
#
#   # Create new matrix with the keywords
#   docs, org_ids = vl.update(docs, keywords)
#
#   # Convert to sparse matrix
#   docs = vl.convert_sparse(docs, keywords)
#
#   # Clustering
#   print()
#   print("Clustering...")
#
#   # n_components: number of dimensions for LSA
#   # k: number of clusters
#   # n_desc: number of keywords (desc) for each cluster
#
#   if dataset == "PLOS":
#       membership, cluster_centers, cluster_desc, coordinates, error = \
#           vl.kmeans(docs, keywords, n_components=50, k=num_cls,
#                     n_desc=25)
#   else:
#       membership, cluster_centers, cluster_desc, coordinates, error = \
#           vl.kmeans(docs, keywords, n_components=20, k=num_cls,
#                     n_desc=15)
#
#   if error != None:  # needs to be changed
#       # flask(error)
#       return redirect(url_for('home'))
#
#   # get cluster colors
#   '''
#   # center is (.5,0)
#   cc = np.array(cluster_centers)-[.5,0]
#   hue = ((np.arctan2(cc[:,1],cc[:,0])/np.pi+1)*180*2).\
#       astype(int).tolist()
#   satr = (np.linalg.norm(cc, axis=1)/math.sqrt(5)*2).tolist()
#   '''
#   # center is (.5,.5)
#   cc = np.array(cluster_centers)-.5
#   hue = ((np.arctan2(cc[:, 1], cc[:, 0])/np.pi+1)*180).\
#       astype(int).tolist()
#   satr = (np.linalg.norm(cc, axis=1)/math.sqrt(2)*2).tolist()
#   val = [.8] * num_cls
#
#   # Create data to pass to result.html
#   df_new = dict()
#   dfr_new = dict()
#   for w in keywords:
#       df_new[w] = df[w]
#       dfr_new[w] = dfr[w]
#
#   # count cluster members
#   id2members = defaultdict(list)
#   for i, m in enumerate(membership):
#       id2members[m].append(i)
#
#   # create adjacency matrix that will be used for network edges
#   centroid_matrix = pd.DataFrame.from_records(cluster_centers)
#   centroid_distances = pd.DataFrame(squareform(pdist(
#       centroid_matrix, metric='euclidean')), columns=range(num_cls), index=range(num_cls)).to_dict()
#
#   '''
#   id2freq = dict()
#   for k in id2members.keys():
#       id2freq[k] = len(id2members[k])
#   '''
#
#   id2freq = {x: len(id2members[x]) for x in range(len(id2members))}
#
#   membership = np.array(membership)
#
#   # create network data structure
#   print('\nEUCLIDEAN DISTANCES\n----------')
#   edges = []
#   for i in range(num_cls):
#       for k, v in centroid_distances[i].items():
#           print(i, ' to ', k, ' = ', v)
#           if i != k:
#               if v > 0.75:
#                   v = "Distant"
#                   edge = {
#                       "clusterID": i, "source": cluster_centers[i], "target": cluster_centers[k], "distance": v}
#                   edges.append(edge)
#               elif v <= 0.75 and v > 0.50:
#                   v = "Similar"
#                   edge = {
#                       "clusterID": i, "source": cluster_centers[i], "target": cluster_centers[k], "distance": v}
#                   edges.append(edge)
#               elif v <= 0.50:
#                   v = "Very Similar"
#                   edge = {
#                       "clusterID": i, "source": cluster_centers[i], "target": cluster_centers[k], "distance": v}
#                   edges.append(edge)
#
#               # elif v >= 0.50:
#               #    v = "Not Similar"
#               # edge = {"clusterID":i,"source":cluster_centers[i],"target":cluster_centers[k],"distance":v}
#               # edges.append(edge)
#   print('\nBIBLIOGRAPHY\n----------')
#   sources = []
#   if dataset == "PubMedAPI":
#       for cluster in sorted(id2members.keys()):
#           concepts = cluster_desc[cluster]
#           references = []
#           for bib in id2members[cluster]:
#               references.append(bibs[bib])
#           for paper in references:
#               if type(paper['author']) != str and None in paper['author']:
#                   paper['author'] = [i for i in paper['author'] if i]
#           bibliography = {'concepts': concepts, 'references': references}
#           sources.append(bibliography)
#       print(sources)
#
#   '''
#   cnt = np.unique(membership, return_counts=True)
#   keys = [x for x in cnt[0]]
#   values = [int(x) for x in cnt[1]]
#   id2freq = dict(zip(keys, values))
#   '''
#
#   '''
#   # Prepare bib data if num of documents is small
#   if len(org_ids) > 150:
#       bibs = []
#   '''
#
#   # session['docs'] = docs  # term-doc matrix
#   session['num_cls'] = num_cls
#   session['membership_0'] = membership
#   session['cluster_desc_0'] = cluster_desc
#   session['xy_0'] = cluster_centers
#   session['state'] = state
#   session['hue_0'] = hue
#   session['satr_0'] = satr
#   session['org_ids_0'] = org_ids
#   session['bibs_0'] = bibs
#   session['dataset'] = dataset
#   session['edges_0'] = edges
#   session['sources'] = sources
#
#   # session['df'] = df_new
#   # session['dfr'] = dfr_new
#   # session['keywords'] = keywords
#
#   # print("bibs")
#   # print(bibs)
#   # print(docs)
#
#   # output the data to terminal
#   print('\nCENTROIDS\n----------')
#   print(cluster_centers)
#   print('\nCENTROID DISTANCES\n----------')
#   print(centroid_distances)
#   print('\nCLUSTER LABELS\n----------')
#   print(cluster_desc)
#   print('\nCLUSTER ID & PUBLICATIONS\n----------')
#   print(id2members)
#   print('\nCLUSTER ID & PUBLICATION COUNT\n----------')
#   print(id2freq)
#   print('\nNETWORK\n----------')
#   print(edges)
#
#   # return render_template("result.html",
#   receive_time = time.clock()
#   response_time = str(round(receive_time - send_time, 3))
#   return render_template("results_voice.html",
#                          q=q,
#                          error=error,
#                          keywords=keywords,
#                          df=df_new, dfr=dfr_new,
#                          cluster_desc=cluster_desc,
#                          xy=cluster_centers,
#                          edges=edges,
#                          id2freq=id2freq,
#                          xy_inter=coordinates,
#                          hue=hue,
#                          satr=satr,
#                          val=val,
#                          bibs=bibs,
#                          id2members=id2members,
#                          sources=sources,
#                          dataset=dataset,
#                          response_time=response_time)

# @app.route('/_re_cluster_voice')
# def re_cluster_voice():
#
#   response_time = '0'
#   send_time = time.clock()
#
#   # Get session data
#   state = session['state']
#   num_cls = session['num_cls']
#   dataset = session['dataset']
#   # Get cluster ids
#   ids = request.args.getlist('ids[]')
#   print("ids: ", ids)
#
#   # Get doc ids (in selected clusters)
#   membership = session['membership_'+str(state)]
#   doc_ids = []
#   for id in ids:
#       doc_ids += (membership == int(id)).nonzero()[0].tolist()
#
#   # Get doc data
#   doc_org = session['docs_org']
#   org_ids = session['org_ids_'+str(state)]
#   docs = []
#   org_ids_ = []  # original document ids
#   for id in doc_ids:
#       id_ = org_ids[id]
#       org_ids_.append(id_)
#       docs.append(doc_org[id_])
#
#   # Redo feature selection
#   print("Re-computing TFIDF and finding key terms...")
#   docs, dfr = vl.compute_tfidf(docs, session['df_org'], rank=5)
#
#   # Sort and output results (discovered keywords)
#   keywords = vl.output_keywords(len(docs), dfr,
#                                 session['df_org'], p_docs=0.5)
#
#   # Create new matrix with the keywords
#   docs, org_ids_ = vl.update(docs, keywords, org_ids_)
#   session['org_ids_'+str(state+1)] = org_ids_
#
#   # Prepare bibliographies if num of documents is small
#
#   # if len(org_ids_) < 50:
#   bibs = session['bibs_0']
#   bibs_new = []
#   for id in org_ids_:
#       # print(id, bibs[id])
#       bibs_new.append(bibs[id])
