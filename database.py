import sqlite3
import json

connection = sqlite3.connect("rag_documents.db")
cursor = connection.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        content TEXT,
        embedding TEXT,
        UNIQUE(source, content)
    )
""")


def insert_document(content, embedding, source):
    embedding_json = json.dumps(embedding)

    cursor.execute(
        """
        INSERT OR IGNORE INTO documents (content, embedding, source)
        VALUES (?, ?, ?)
        """,
        (content, embedding_json, source)
    )

    connection.commit()
    return cursor.rowcount  #Returns the number of rows inserted (0 if the document already exists)


def get_documents():
    cursor.execute("SELECT * FROM documents")
    return cursor.fetchall()