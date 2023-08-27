import pandas as pd
import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def load_db(db_path
              , db_name):
    database = "{}{}.db".format(db_path
                                 , db_name)
    conn = create_connection(database)

    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def select_all_from_tbl(conn, table_name):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    # -- set cursor and fetch
    cur = conn.cursor()
    cur.execute("SELECT * FROM "+table_name)
    rows = cur.fetchall()
    # -- Get Data from DB
    df = pd.DataFrame(rows)
    df.columns = [description[0] for description in cur.description]

    return df


def select_daily_from_hourly(conn
                             , table_name):
    # ... Thanks to https://stackoverflow.com/questions/65625544/how-to-query-sqlite-data-hourly
    query = """ SELECT t1.*
    FROM {} t1
    WHERE NOT EXISTS (
      SELECT 1 
      FROM {} t2
      WHERE strftime('%Y%m%d', t2.time) = strftime('%Y%m%d', t1.time)
        AND t2.time > t1.time  
    )
    """.format(table_name, table_name)
    # -- Set up SQL
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    # -- Get Data from DB
    df = pd.DataFrame(rows)
    df.columns = [description[0] for description in cur.description]

    return df

