import math
from foundry_local_sdk import Configuration, FoundryLocalManager

#Knowledge base — each string represents a document
documents = [
    "Foundry Local runs AI models directly on your device without cloud connectivity.",
    "The Foundry Local SDK supports Python, C#, JavaScript, and Rust.",
    "Embedding models convert text into numerical vectors for similarity search.",
    "Foundry Local uses ONNX Runtime for efficient model inference on CPUs and GPUs.",
    "The model catalog provides pre-optimized models that you can download and run locally.",
    "Retrieval-augmented generation grounds model responses in your own data.",
    "Vector similarity search finds documents that are semantically close to a query.",
    "Chat completions generate natural language responses from a prompt and context.",
]

def main():
    #Initialize the SDK
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    #Load the embedding model
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    #Embed all documents in a single batch call
    response = embedding_client.generate_embedding(documents)
    doc_embeddings = [item.embedding for item in response.data]
    print(f"Indexed {len(doc_embeddings)} documents.")

    def cosine_similarity(a, b):
        #Compute cosine similarity between two vectors (RAG should find similar texts)
        dot = sum(x * y for x, y in zip(a, b)) #dot product of these vectors
        norm_a = math.sqrt(sum(x * x for x in a)) #length of vector a
        norm_b = math.sqrt(sum(x * x for x in b)) #length of vector b
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0 #cosine similarity
        #we need to divide because length of vectors affect dot product

    def find_relevant(query_embedding, doc_embeddings, top_k=2):
        #Return the indices and scores of the top-k most similar documents.
        #query_embeddings: user embeds, doc_embeddings = document embeds, top_k=2 best two documents
        scores = []
        for i, doc_emb in enumerate(doc_embeddings):
            score = cosine_similarity(query_embedding, doc_emb)
            scores.append((i, score)) 
        scores.sort(key=lambda x: x[1], reverse=True) #sort scores descending by scores
        return scores[:top_k]
