import math
from foundry_local_sdk import Configuration, FoundryLocalManager
from pathlib import Path
from database import insert_document, get_documents
import json

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

def get_top_chunks(query, embedding_client, top_k=2):
    # Embed the user's query
    response = embedding_client.generate_embedding(query)
    query_embedding = response.data[0].embedding

    # Get all stored documents from SQLite
    documents = get_documents()

    # Convert stored embeddings from JSON strings back to Python lists
    chunk_embeddings = [
        json.loads(document[3])
        for document in documents
    ]

    # Find the most similar chunks
    top_results = find_relevant(
        query_embedding,
        chunk_embeddings,
        top_k
    )

    results = []

    for index, score in top_results:
        document = documents[index]

        results.append({
            "source": document[1],
            "content": document[2],
            "score": score
        })

    return results

def answer_query(query, embedding_client, chat_client):
    #Retrieve the top relevant chunks based on the query
    results = get_top_chunks(query, embedding_client)

    #Build the context from the retrieved chunks
    context = "\n\n".join(
        f"Source: {result['source']}\n{result['content']}"
        for result in results
    )

    #Create the messages for the chat model
    messages = [
        {
            "role": "system",
            "content": (
                "You are a question-answering assistant for a local knowledge base.\n"
                "You MUST answer using only the information provided in the context.\n"
                "Do NOT use your own knowledge or make assumptions.\n"
                "If the answer is not explicitly stated in the context, "
                "respond exactly with: \"I don't know based on the provided context.\"\n"
                "When you answer a question, always include the source document name(s) "
                "at the end of your answer in the format: 'Sources: filename.txt'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )
        }
    ]

    # Generate the answer
    response = chat_client.complete_chat(messages)
    return response.choices[0].message.content



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

    #------Download the embedding model--------
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()

    #Load the embedding model and get the embedding client
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    #------Download the chat model--------
    chat_model = manager.catalog.get_model("phi-4-mini")
    chat_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )
    print()

    #Load the chat model and get the chat client
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    #Embed the chunks and store the embeddings in a list
    texts = [chunk["content"] for chunk in chunks]
    chunk_embeddings = []

    for text in texts:
        response = embedding_client.generate_embedding(text)
        chunk_embeddings.append(response.data[0].embedding)

    print(f"Embedded {len(chunk_embeddings)} chunks.")

    #Store the chunks and their embeddings in the database
    inserted_count = 0 #Count of new chunks inserted into the database

    for chunk, embedding in zip(chunks, chunk_embeddings):
        inserted_count += insert_document( #Store the chunk in the database, if it already exists, it will not be inserted again
            chunk["content"],
            embedding,
            chunk["source"]
        )

    print(f"Inserted {inserted_count} new chunks into the database.")

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        answer = answer_query(query, embedding_client, chat_client)
        print("\nAnswer:")
        print(answer)
        if query.lower() == "exit":
            print("Exiting the program.")
            break

    

        

if __name__ == "__main__":
    main()



