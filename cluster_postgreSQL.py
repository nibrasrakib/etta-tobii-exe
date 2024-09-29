import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def execute_query(query, hostname, port, database, username, password):
    error = None
    # Database connection parameters
    # hostname = 'localhost'
    # port = 5433
    # database = 'PATTIE'
    # username = 'postgres'
    # password = 'Arvi1308'  # replace with your actual password
    # hostname = '34.133.177.246'
    # port = 5432
    # database = 'rss_feed'
    # username = 'aravind'
    # password = 'C&99Fk6xHxypA2R$C4XQ'
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

# Function to search the database by keyword in title or description
def search_database_by_keyword(keyword, 
                                hostname = '34.133.177.246',
                                port = 5432,
                                database = 'rss_feed',
                                username = 'aravind',
                                password = 'C&99Fk6xHxypA2R$C4XQ'):
    try:
        # Connect to the database
        connection = psycopg2.connect(host=hostname, port=port, dbname=database, user=username, password=password)
        
        # Create a cursor
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Format the keyword for full-text search
        formatted_keyword = ' & '.join(keyword.split())

        # Execute the query
        cursor.execute("""
            SELECT * FROM rss_feed
            WHERE to_tsvector('english', title || ' ' || description) @@ to_tsquery('english', %s);
            """, (formatted_keyword,))
        
        # Fetch all the results
        results = cursor.fetchall()

        # Convert the results to a DataFrame
        results = pd.DataFrame(results)
        
        # Close the cursor and connection
        cursor.close()
        connection.close()

        return results, None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], e
    
    
def optimal_number_of_clusters(data, max_clusters=10):
    """
    Determine the optimal number of clusters using the Elbow method and silhouette score.
    """
    inertia = []
    silhouette_scores = []
    K = range(2, max_clusters+1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)  # Sum of squared distances to the nearest cluster center
        
        # Calculate silhouette score only if k > 1
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, 'bo-', label='Inertia')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.legend()
    plt.show()

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'ro-', label='Silhouette Score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for different number of clusters')
    plt.legend()
    plt.show()

    # Find the optimal number of clusters based on silhouette score
    best_k = np.argmax(silhouette_scores) + 2  # because we started with k=2

    return best_k