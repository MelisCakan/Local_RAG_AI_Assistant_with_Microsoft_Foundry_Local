import math
from foundry_local_sdk import Configuration, FoundryLocalManager
from pathlib import Path

#Load documents from a folder and return a list of dictionaries with source and content
def load_documents(folder_path):
    documents = []

    for file in Path(folder_path).glob("*.txt"): #find all txt files in the folder
        with open(file, "r", encoding="utf-8") as f:
            documents.append({
                "source": file.name, #name of the file as source
                "content": f.read() #read the content of the file
            })

    return documents

#Chunk the documents into paragraphs and return a list of dictionaries with source and content
def chunk_documents(documents): 
    chunks = []

    for document in documents:
        paragraphs = document["content"].split("\n\n") #split content into paragraphs

        for paragraph in paragraphs:
            paragraph = paragraph.strip() #remove leading and trailing whitespace

            if paragraph: #if paragraph is not empty
                chunks.append({
                    "source": document["source"],
                    "content": paragraph
                })

    return chunks

def cosine_similarity(a, b):
        #Compute cosine similarity between two vectors (RAG should find similar texts)
        dot = sum(x * y for x, y in zip(a, b)) #dot product of these vectors
        norm_a = math.sqrt(sum(x * x for x in a)) #length of vector a
        norm_b = math.sqrt(sum(x * x for x in b)) #length of vector b
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0 #cosine similarity
        #we need to divide because length of vectors affect dot product

def find_relevant(query_embedding, chunk_embeddings, top_k=2):
    #Return the indices and scores of the top-k most similar chunks.
    #query_embeddings: user embeds, chunk_embeddings = chunk embeds, top_k=2 best two chunks
    scores = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_emb)
        scores.append((i, score)) 
    scores.sort(key=lambda x: x[1], reverse=True) #sort scores descending by scores
    return scores[:top_k]


def main():

    #Load documents from the "documents" folder
    documents = load_documents("docs")
    print(f"Loaded {len(documents)} documents.")

    #Chunk the documents into paragraphs
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

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

    #Load the embedding model and get the embedding client
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    #Embed the chunks and store the embeddings in a list
    texts = [chunk["content"] for chunk in chunks]
    chunk_embeddings = []

    for text in texts:
        response = embedding_client.generate_embedding(text)
        chunk_embeddings.append(response.data[0].embedding)

    print(f"Embedded {len(chunk_embeddings)} chunks.")

if __name__ == "__main__":
    main()



