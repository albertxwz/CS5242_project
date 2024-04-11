import sqlite3
 
DB_NAME = 'application-db.db'

def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("CREATE TABLE user (U_ID integer primary key, U_NAME text);")
    conn.commit()
    
    return 200
 

def insert_content():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO user values('100', '200')")
    conn.commit()
    
    return 200


def get_content():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM user")
    print(cursor.fetchall())
    
    return 200



# create_table()
# insert_content()
get_content()
    