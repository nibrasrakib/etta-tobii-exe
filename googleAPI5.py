#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Michael Segundo Ortiz
Date: 1/27/20
Project: Google API search for CDC, Medline Plus, WHO (Global Public Health)
Documentation on API used: https://developers.google.com/custom-search/v1/overview
Custom search ID – 003950716982973815356:wa5gmnqtvso
Public URL - https://cse.google.com/cse?cx=003950716982973815356:wa5gmnqtvso
NOTE: start parameter is the result number not page number – https://developers.google.com/custom-search/v1/cse/list#request
"""
from googleapiclient.discovery import build
import pandas as pd

def retrieve(q):

		query = q
		error = None
		df = None
		# currently free tier – 100 queries per day
		# pip install google-api-python-client
		api_key='AIzaSyAPjEL_6vmlQRYRKgOj8yeNDM11tBCN9KQ'
		# custom search engine only retrieves *.edu domains
		cse_id = '003950716982973815356:wa5gmnqtvso'
		if query == '':
				error = 'Please provide search terms'
		else:
				service = build("customsearch", "v1", developerKey=api_key)
				# API limit of also 100 results per search
				page_iterator = [1,11,21,31,41,51,61,71,81,91]
				result_count = 0
				df = pd.DataFrame()
				for i in page_iterator:
						result = service.cse().list(q=query, num=10, start=i, cx=cse_id).execute()
						try:
								# metadata = result['searchInformation']
								data = result['items']
								for info in data:
										title = info['title']
										url = info['link']
										snippet = info['snippet']
										result_count += 1
										# create a clean dictionary with elements
										data = {'title':title,
														'url': url,
														'snippet': snippet}
										# build series object and build a dataframe for analysis later
										series = pd.Series(data)
										df = df.append(series,ignore_index=True)
						except:
								return df, error
				return df, error
