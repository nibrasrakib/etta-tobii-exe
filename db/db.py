import traceback
import mysql.connector

# from mysql.connector import Error

# pip3 install mysql-connector
# https://dev.mysql.com/doc/connector-python/en/connector-python-reference.html


class DB:
    def __init__(self, config):
        self.connection = None
        self.connection = mysql.connector.connect(**config)

    def query(self, sql, args):
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(sql, args)
        except:
            print(cursor.statement)
            print(traceback.format_exc())
        return cursor

    def insert(self, sql, args):
        cursor = self.query(sql, args)
        id = cursor.lastrowid
        self.connection.commit()
        cursor.close()
        return id

    # https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-executemany.html
    def insertmany(self, sql, args):
        cursor = self.connection.cursor()
        cursor.executemany(sql, args)
        rowcount = cursor.rowcount
        self.connection.commit()
        cursor.close()
        return rowcount

    def update(self, sql, args):
        cursor = self.query(sql, args)
        rowcount = cursor.rowcount
        self.connection.commit()
        cursor.close()
        return rowcount

    def fetch(self, sql, args):
        rows = []
        cursor = self.query(sql, args)
        print(cursor.statement)
        if cursor.with_rows:
            rows = cursor.fetchall()
        cursor.close()
        return rows

    def fetchone(self, sql, args):
        row = None
        cursor = self.query(sql, args)
        if cursor.with_rows:
            row = cursor.fetchone()
        cursor.close()
        return row

    def __del__(self):
        print("db connection closed")
        if self.connection != None:
            self.connection.close()
