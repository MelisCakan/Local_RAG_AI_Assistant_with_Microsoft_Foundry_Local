import json
import sqlite3


def get_connection():
    connection = sqlite3.connect("rag_documents.db")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            content TEXT,
            embedding TEXT,
            UNIQUE(source, content)
        )
        """
    )
    return connection


def insert_document(content, embedding, source):
    embedding_json = json.dumps(embedding)

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO documents (content, embedding, source)
            VALUES (?, ?, ?)
            """,
            (content, embedding_json, source),
        )
        connection.commit()
        return cursor.rowcount


def get_documents():
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM documents")
        return cursor.fetchall()


def get_document_sources():
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT source FROM documents ORDER BY source")
        return [row[0] for row in cursor.fetchall()]


def delete_document_by_source(source):
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM documents WHERE source = ?", (source,))
        connection.commit()
        return cursor.rowcount


def delete_document_by_id(document_id):
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        connection.commit()
        return cursor.rowcount
