import sqlite3

connection = sqlite3.connect("rag_documents.db")
cursor = connection.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, embedding BLOB)")

def insert_document(content, embedding):
    cursor.execute("INSERT INTO documents (content, embedding) VALUES (?, ?)", (content, embedding))
    connection.commit()

def get_documents():
    cursor.execute("SELECT * FROM documents")
    return cursor.fetchall()

