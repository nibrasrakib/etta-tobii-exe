# -*- coding: utf-8 -*-
"""
Author: Michael Segundo Ortiz
Date: 6/15/19
Project: Cluster News
Documentation on API used: https://newsapi.org

"""
# import necessary libraries
from newsapi import NewsApiClient as news
from datetime import datetime
from datetime import timedelta
import pandas as pd


# API key recieved on 6/15/2019 from https://newsapi.org/register/success
api_key = 'ec70ad2f634a4f17a6938b2d90503c2d'
'''
def main():
    df = retrieve('bitcoin')
    data = organize(df)
    print(data['description'])
'''
# this block of code will retrieve top headlines across > 50,000 thousands sources
def retrieve(query, start_time, end_time):
    newsapi = news(api_key=api_key)
    if query == '':
        all_articles = newsapi.get_top_headlines(language='en',
                                                 page_size=100)
        df = pd.DataFrame.from_dict(all_articles)
        return df

##### UNCOMMENT THIS BLOCK IF YOU WANT A "SEARCH" FUNCTION" #####
    #query = input('bitcoin')
    else:
        today = datetime.today().strftime('%Y-%m-%d')
        if end_time == '':
            end_time = today
        #yesterday = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
        all_articles = newsapi.get_everything(q=query,
                                          from_param=start_time,
                                          to=end_time,
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=100)
        df = pd.DataFrame.from_dict(all_articles)
        return df

# this block of code will organize the retrieved headlines for indexing
def organize(df):
    source = []
    author = []
    title = []
    description = []
    content = []
    url = []
    articles = df['articles']
    article_dict = articles.to_dict()
    for index, article in article_dict.items():
        source.append(article['source']['name'])
        author.append(article['author'])
        title.append(article['title'])
        description.append(article['description'])
        content.append(article['content'])
        url.append(article['url'])
    # create an organized dicitonary and dataframe from data elements
    organized = {'source':source,
                 'author':author,
                 'title':title,
                 'description':description,
                 'content':content,
                 'url':url}
    data = pd.DataFrame(data=organized)
    return data

# run the main function to start cluster news
#main()
