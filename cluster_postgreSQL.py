import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor

def execute_query(query, hostname, port, database, username, password):
    error = None
    # Database connection parameters
    # hostname = 'localhost'
    # port = 5433
    # database = 'PATTIE'
    # username = 'postgres'
    # password = 'Arvi1308'  # replace with your actual password
    if query is None:
        error = "Query is empty"
        return [], error
    try:
        # Establishing the connection
        conn = psycopg2.connect(
            host=hostname,
            port=port,
            dbname=database,
            user=username,
            password=password
        )

        # Create a cursor object using RealDictCursor to get results as dictionaries
        # cur = conn.cursor(cursor_factory=RealDictCursor)
        cur = conn.cursor()

        # Execute the query
        cur.execute(query)

        # Fetch all the results
        results = cur.fetchall()

        # Close the cursor and the connection
        cur.close()
        conn.close()

        # Return the results
        return results, error

    except Exception as e:
        error = f"Unexpected error: {e}"
        print(f"An error occurred: {e}")
        if conn:
            conn.close()
        return [], error


def retrieve_from_postgresql(author_name):
    hostname = '34.133.177.246'
    port = 5432
    database = 'dcdi'
    username = 'aravind'
    password = 'C&99Fk6xHxypA2R$C4XQ'

    # query = f"""
    #     SELECT title, abstract, authors, url
    #     FROM papers
    #     WHERE authors::jsonb @> '[\"{author_name}\"]';
    #     """
    query = f"""
        SELECT title, abstract, authors, url
        FROM papers, jsonb_array_elements_text(authors) as author
        WHERE author ILIKE '{author_name}%'  -- Matches names that start with {author_name}
        OR author ILIKE '% {author_name}' -- Matches names that end with {author_name}
        OR author ILIKE '% {author_name} %'; -- Matches names that have {author_name} as a full word
        """
    rows, error = execute_query(query, hostname, port, database, username, password)
    print("ROWS:")
    print(rows)
    if rows:
        # Process each row to ensure consistent structure
        processed_rows = []
        for row in rows:
            # Debugging: Print the structure of each row
            print("Row structure:", type(row), row)
            
            # Each row is a tuple, ensure it has all columns, replace None with empty string
            processed_row = tuple(item if item is not None else '' for item in row)
            processed_rows.append(processed_row)
        
        if processed_rows:
            # Define columns as per the table structure
            columns = [
                'title', 'abstract', 'authors', 'url'
            ]
            # Create a DataFrame from the processed results
            df = pd.DataFrame(processed_rows, columns=columns)
    else:
        df = pd.DataFrame()
    
    return df, error