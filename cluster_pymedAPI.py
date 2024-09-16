#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Michael Segundo Ortiz
Date: 6/15/19
Project: Cluster pymed
Documentation on API used: https://pypi.org/project/pymed/

"""
from pymed import PubMed
import spacy
import pandas as pd
import traceback
import json
import xml.etree.ElementTree as ET


def main():
    df, err = retrieve("Fei Yu")
    print(df.columns)


def retrieve(q):
    print("Retrieving data from PubMed")
    # initialize the class PubMed
    pubmed = PubMed(tool="PATTIE", email="nibras.rakib@mail.utoronto.ca")
    query = q
    error = None
    df = None
    if query == "":
        error = "Please provide search terms"
    else:
        # query PubMed object with user input
        try:
            results = pubmed.query(query, max_results=300)
        except:
            print("Unexpected error: ", traceback.print_exc())
            raise
        df = pd.DataFrame()
        pubmed_dict = {}
        for result in results:
            # convert retrieved object to dictionary structure
            pubmed_dict = result.toDict()
            # xml_string = ET.tostring(
            #     pubmed_dict["xml"], encoding="unicode", method="xml"
            # )
            # print(xml_string)
            print("Size of pubmed_dict: ", len(pubmed_dict))
            print("Keys of pubmed_dict: ", pubmed_dict.keys())
            if "xml" not in pubmed_dict.keys():
                continue
            print("pubmed_dict: ", pubmed_dict["xml"])
            # Print every it can .find in the xml
            print(
                "MedlineCitation: ",
                pubmed_dict["xml"].find("MedlineCitation").find("Article").find("Journal").find("JournalIssue").find("PubDate").find("Year"),
            )
            xml_root = pubmed_dict["xml"]
            xml_string = ET.tostring(xml_root, encoding="unicode", method="xml")
            if pubmed_dict["xml"].find("MedlineCitation").find("Article").find("Journal").find("JournalIssue").find("PubDate").find("Year") is None:
                continue
            pubYear = (
                xml_root.find("MedlineCitation")
                .find("Article")
                .find("Journal")
                .find("JournalIssue")
                .find("PubDate")
                .find("Year")
                .text
            )
            mesh_descriptor_names = [
                {"descriptor": descriptor.text}
                for descriptor in xml_root.findall(".//DescriptorName")
            ]
            # print(mesh_descriptor_names)
            # extract elements
            authors = []
            affiliations = []
            for a in pubmed_dict["authors"]:
                try:
                    firstname = a["firstname"]
                    lastname = a["lastname"]
                    affiliation = a["affiliation"]
                    fullname = firstname + " " + lastname
                    authors.append(fullname)
                    affiliations.append(affiliation)
                except:
                    lastname = a["lastname"]
                    authors.append(lastname)
            if None in authors:
                authors.remove(None)
            affiliations = [i for i in affiliations if i is not None]
            title = pubmed_dict["title"]
            abstract = pubmed_dict["abstract"]
            pmid = pubmed_dict["pubmed_id"]
            pubdate = pubmed_dict["publication_date"]
            # if statement handles multiple PMIDs and only retrieves the first one (correct one)
            if "\n" in pmid:
                pmid_list = pmid.split("\n")
                pmid = pmid_list[0]
            url = "https://www.ncbi.nlm.nih.gov/pubmed/?term=" + pmid
            # create a clean dictionary with elements
            data = {
                "authors": authors,
                "affiliations": affiliations,
                "title": title,
                "abstract": abstract,
                "pmid": pmid,
                "url": url,
                "pubYear": pubYear,
                "pubdate": pubdate,
                "meshHeadings": mesh_descriptor_names,
            }
            # print(data)
            # build series object and build a dataframe for analysis later
            series = pd.Series(data)
            # df = df.append(series, ignore_index=True)
            series_df = pd.DataFrame([series])
            # Use pandas.concat to append the series to the DataFrame
            df = pd.concat([df, series_df], ignore_index=True)
    print(df.get("authors"), type(df.get("authors")))
    print("Going to return df")
    return df, error


# main()
