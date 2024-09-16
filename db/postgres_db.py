import psycopg2
from psycopg2 import sql, extras


class PostgresDB:
    def __init__(self, config):
        self.conn = psycopg2.connect(**config)
        self.cur = self.conn.cursor(cursor_factory=extras.RealDictCursor)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def insert(self, table, columns, values):
        query = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({values})").format(
            table=sql.Identifier(table),
            columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
            values=sql.SQL(", ").join(sql.Placeholder() * len(values)),
        )
        self.cur.execute(query, values)
        last_row_id = self.cur.fetchone()[0]
        self.conn.commit()
        return last_row_id

    def insert_many(self, table, columns, values_list):
        query = sql.SQL("INSERT INTO {table} ({columns}) VALUES %s").format(
            table=sql.Identifier(table),
            columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
        )
        extras.execute_values(self.cur, query, values_list)
        row_count = self.cur.rowcount
        self.conn.commit()
        return row_count

    def update(self, table, set_columns, set_values, condition_column, condition_value):
        set_clause = sql.SQL(", ").join(
            sql.SQL("{0} = {1}").format(sql.Identifier(col), sql.Placeholder())
            for col in set_columns
        )
        query = sql.SQL(
            "UPDATE {table} SET {set_clause} WHERE {condition_column} = {condition_value}"
        ).format(
            table=sql.Identifier(table),
            set_clause=set_clause,
            condition_column=sql.Identifier(condition_column),
            condition_value=sql.Placeholder(),
        )
        self.cur.execute(query, set_values + [condition_value])
        self.conn.commit()

    def fetch(self, table, columns, condition=None):
        if condition:
            query = sql.SQL("SELECT {columns} FROM {table} WHERE {condition}").format(
                columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
                table=sql.Identifier(table),
                condition=sql.SQL(condition),
            )
        else:
            query = sql.SQL("SELECT {columns} FROM {table}").format(
                columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
                table=sql.Identifier(table),
            )
        self.cur.execute(query)
        return self.cur.fetchall()

    def fetch_one(self, table, columns, condition=None):
        if condition:
            query = sql.SQL(
                "SELECT {columns} FROM {table} WHERE {condition} LIMIT 1"
            ).format(
                columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
                table=sql.Identifier(table),
                condition=sql.SQL(condition),
            )
        else:
            query = sql.SQL("SELECT {columns} FROM {table} LIMIT 1").format(
                columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
                table=sql.Identifier(table),
            )
        self.cur.execute(query)
        return self.cur.fetchone()

    def execute_query(self, query, params=None):
        self.cur.execute(query, params)
        self.conn.commit()
        try:
            return self.cur.fetchall()
        except psycopg2.ProgrammingError:  # No result to fetch
            return None

    def close(self):
        self.cur.close()
        self.conn.close()


# Example usage:
# Example usage with context management:
# with PostgresDB(dbname='yourdbname', user='youruser', password='yourpassword', host='yourhost', port='yourport') as db:
#     last_row_id = db.insert('yourtable', ['column1', 'column2'], ['value1', 'value2'])
#     print("Last inserted row ID:", last_row_id)
#     row_count = db.insert_many('yourtable', ['column1', 'column2'], [('value1', 'value2'), ('value3', 'value4')])
#     print("Number of rows inserted:", row_count)
#     rows = db.fetch('yourtable', ['column1', 'column2'])
#     print(rows)
