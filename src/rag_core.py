import json
import math
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

from src.database import (
    delete_document_by_source,
    get_document_sources,
    get_documents,
    insert_document,
)


def load_documents(folder_path):
    documents = []

    for file in Path(folder_path).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            documents.append({
                "source": file.name,
                "content": f.read(),
            })

    return documents


def chunk_documents(documents):
    chunks = []

    for document in documents:
        paragraphs = document["content"].split("\n\n")

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                chunks.append({
                    "source": document["source"],
                    "content": paragraph,
                })

    return chunks


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def find_relevant(query_embedding, chunk_embeddings, top_k=2):
    scores = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_emb)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def get_top_chunks(query, embedding_client, top_k=2):
    response = embedding_client.generate_embedding(query)
    query_embedding = response.data[0].embedding
    documents = get_documents()

    chunk_embeddings = [json.loads(document[3]) for document in documents]

    if not chunk_embeddings:
        return []

    top_results = find_relevant(query_embedding, chunk_embeddings, top_k)
    results = []

    for index, score in top_results:
        document = documents[index]
        results.append({
            "source": document[1],
            "content": document[2],
            "score": score,
        })

    return results


def initialize_models():
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    chat_model = manager.catalog.get_model("phi-4-mini")
    chat_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    return embedding_client, chat_client


def ingest_document_file(file_path, embedding_client):
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_documents([{"source": path.name, "content": content}])
    if not chunks:
        return 0

    inserted_count = 0
    for chunk in chunks:
        response = embedding_client.generate_embedding(chunk["content"])
        embedding = response.data[0].embedding
        inserted_count += insert_document(
            chunk["content"],
            embedding,
            chunk["source"],
        )

    return inserted_count


def delete_document(source_name):
    return delete_document_by_source(source_name)


def list_uploaded_documents():
    return get_document_sources()


def answer_query(query, embedding_client, chat_client):
    if not query or not query.strip():
        return "Please enter a question."

    results = get_top_chunks(query, embedding_client)

    if not results:
        return "No document content has been uploaded yet. Please add a document first."

    context = "\n\n".join(
        f"Source: {result['source']}\n{result['content']}"
        for result in results
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a question-answering assistant for a local knowledge base.\n"
                "You MUST answer using only the information provided in the context.\n"
                "Do NOT use your own knowledge or make assumptions.\n"
                "If the answer is not explicitly stated in the context, "
                "respond exactly with: \"I don't know based on the provided context.\"\n"
                "IMPORTANT: If you respond with \"I don't know based on the provided context.\", "
                "DO NOT include any sources.\n"
                "For valid answers, always include the source document name(s) "
                "at the end of your answer in the format: 'Sources: filename.txt'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            ),
        },
    ]

    response = chat_client.complete_chat(messages)
    answer = response.choices[0].message.content

    if "I don't know based on the provided context." in answer:
        return "I don't know based on the provided context."

    return answer