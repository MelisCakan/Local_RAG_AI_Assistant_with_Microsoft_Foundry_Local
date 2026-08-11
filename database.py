import sqlite3

connection = sqlite3.connect("rag_documents.db")
cursor = connection.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, content TEXT, embedding TEXT)")

def insert_document(content, embedding, source):
    cursor.execute("INSERT INTO documents (content, embedding, source) VALUES (?, ?, ?)", (content, embedding, source))
    connection.commit()

def get_documents():
    cursor.execute("SELECT * FROM documents")
    return cursor.fetchall()

