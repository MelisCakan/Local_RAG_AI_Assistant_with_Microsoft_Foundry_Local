import os

from database import (
    delete_document_by_source,
    get_document_sources,
    insert_document,
)


def test_delete_document_by_source_removes_all_chunks_for_source():
    source_name = "test_source.txt"

    insert_document("first chunk", [0.1, 0.2], source_name)
    insert_document("second chunk", [0.3, 0.4], source_name)
    insert_document("other chunk", [0.5, 0.6], "another_source.txt")

    delete_document_by_source(source_name)

    sources = get_document_sources()
    assert source_name not in sources
    assert "another_source.txt" in sources

    os.remove("rag_documents.db")
