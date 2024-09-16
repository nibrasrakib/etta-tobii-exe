import traceback
from flask import current_app
import pandas as pd
from db.postgres_db import PostgresDB as DB
import json


def get_dc_data(q):
    df = None
    error = None
    try:
        db_config = current_app.config["DB_CONFIG"]
        db = DB(db_config)
        sql = """
        SELECT  
            r.title, r.date_created as "pubdate",date_part('year', r.date_created) as "pubYear", REGEXP_REPLACE(
    REGEXP_REPLACE(r.description, '<jats:[^>]*>', ''),
    '</jats:[^>]*>',
    ''
) as abstract
            , r.description as original_text
            , array_to_json(array_agg(trim(c.creator_name))) as authors, '' as affiliation,'' as meshheadings,
            concat('https://doi.org/', i.identifier_value) as url,
            i.identifier_value as pmid
            FROM resource r
            join creator c on r.resource_id = c.resource_id
            join identifier i on r.resource_id = i.resource_id
            where to_tsvector(r.title) || ' ' || to_tsvector(r.description) @@ phraseto_tsquery('{q}')
            group by r.resource_id, r.title, r.description, r.source,i.identifier_value, r.type
            having sum(LENGTH(r.description) - LENGTH(REPLACE(r.description, ' ', '')) + 1)>100
            limit 300
            ;
        """.format(
            q=q
        )
        df = pd.DataFrame()
        rows = db.execute_query(sql)
        for i, item in enumerate(rows):
            authors = []
            print(item["authors"], type(item["authors"]), item["meshheadings"])
            authors = item["authors"]
            data = {
                "authors": authors,
                "affiliations": item["affiliation"],
                "title": item["title"],
                "abstract": item["abstract"],
                "pmid": item["pmid"],
                "url": item["url"],
                "pubYear": item["pubYear"],
                "pubdate": item["pubdate"],
                "meshHeadings": item["meshheadings"],
            }
            series = pd.Series(data)
            df = df.append(series, ignore_index=True)
        print(df.get("authors"))

    except Exception as ex:
        print("DC DATA RETRIEVE ERROR", traceback.print_exc())
        error = "Error"
    return df, error


def main():
    get_dc_data("Javed Mostafa")


# main()
