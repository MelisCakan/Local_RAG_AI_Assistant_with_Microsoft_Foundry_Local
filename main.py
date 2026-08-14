from src.rag_core import (
    answer_query,
    chunk_documents,
    delete_document,
    initialize_models,
    ingest_document_file,
    list_uploaded_documents,
    load_documents,
)

from src.database import insert_document


def main():
    documents = load_documents("docs")
    print(f"Loaded {len(documents)} documents.")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    embedding_client, chat_client = initialize_models()

    chunk_embeddings = []
    for chunk in chunks:
        response = embedding_client.generate_embedding(chunk["content"])
        chunk_embeddings.append(response.data[0].embedding)

    print(f"Embedded {len(chunk_embeddings)} chunks.")

    inserted_count = 0
    for chunk, embedding in zip(chunks, chunk_embeddings):
        inserted_count += insert_document(
            chunk["content"],
            embedding,
            chunk["source"],
        )

    print(f"Inserted {inserted_count} new chunks into the database.")

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting the program.")
            break

        answer = answer_query(query, embedding_client, chat_client)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()

