# Local RAG AI Assistant with Microsoft Foundry Local

## Project Purpose

This project is a local document Q&A assistant developed using **Microsoft Foundry Local** and **Retrieval-Augmented Generation (RAG)**.

The assistant allows users to ask questions about a local collection of documents. Instead of relying only on the language model's existing knowledge, the system retrieves relevant information from the local knowledge base and provides it as context to a local LLM. This helps produce more accurate and source-grounded responses while reducing the risk of unsupported answers.

### Key Features

- 📚 **Document Management** — Upload, store, and manage `.txt` documents
- 🔍 **Semantic Search** — Retrieve relevant content using vector embeddings
- 💬 **Local LLM** — Generate answers using a locally running language model
- 🎨 **User-Friendly GUI** — Manage documents and ask questions through a desktop interface
- ⚡ **Offline Operation** — AI processing can run locally without cloud-based LLM services
- 🔐 **Data Privacy** — Documents and queries remain on the local machine

---

## How It Works

The system follows a Retrieval-Augmented Generation pipeline:

1. **Document Ingestion**  
   Documents are loaded and split into paragraph-based chunks.

2. **Embedding Generation**  
   Each chunk is converted into an embedding using **Qwen3-Embedding-0.6B**.

3. **Database Storage**  
   Document chunks, embeddings, and source filenames are stored in a local **SQLite** database.

4. **Retrieval**  
   When a user asks a question, the query is embedded and compared with stored embeddings using **cosine similarity**. The two most relevant chunks are retrieved.

5. **Answer Generation**  
   The retrieved content is provided as context to **Phi-4-mini**. The model is instructed to answer only from the provided context and include the source documents.

If the required information is not available, the assistant responds:

> `I don't know based on the provided context.`

---

## Project Structure

```text
Local RAG AI Assistant with Microsoft Foundry Local/
│
├── README.md
├── main.py                  # CLI interface
├── app.py                   # CustomTkinter GUI
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── rag_core.py          # RAG pipeline and model operations
│   └── database.py          # SQLite database operations
│
├── docs/                    # Knowledge base documents
│   ├── artificial_intelligence.txt
│   ├── deep_learning.txt
│   ├── machine_learning.txt
│   ├── natural_language_processing.txt
│   └── neural_networks.txt
│
└── tests/
    ├── test_rag.py          # Automated functional tests
    └── test_cases.json      # Test cases
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MelisCakan/Local_RAG_AI_Assistant_with_Microsoft_Foundry_Local.git
cd "Local RAG AI Assistant with Microsoft Foundry Local"
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

The required local models are downloaded automatically when the application initializes.

```bash
python app.py
```

The models are cached locally and reused on subsequent runs.

---

## 🚀 Usage

### GUI Application

Run:

```bash
python app.py
```

The GUI allows users to:

- Upload `.txt` documents
- View uploaded documents
- Delete documents
- Ask questions about the knowledge base
- View answers and source documents

### Command-Line Interface

Alternatively:

```bash
python main.py
```

The CLI loads documents from the `docs/` folder and provides an interactive question-answering interface.

Example:

```text
Enter your question (or type 'exit' to quit): What is machine learning?

Answer:
Machine learning is a subset of artificial intelligence...
Sources: machine_learning.txt
```

---

## 🔧 Design Decisions

### Local Models

**Qwen3-Embedding-0.6B** is used for embedding generation and **Phi-4-mini** for answer generation. Both models run locally through Microsoft Foundry Local.

### SQLite

SQLite was selected because the project uses a relatively small local knowledge base and does not require an external database server.

### Retrieval

The system uses **cosine similarity** and retrieves the **top 2 chunks** for each query. Retrieving multiple chunks allows the model to combine information from different parts of the knowledge base.

### Grounded Responses

The system prompt instructs the model to use only the retrieved context and return a fallback response when the required information is not available. This reduces the risk of unsupported or hallucinated answers.

---

## 🧪 Testing & Evaluation

The project includes an automated functional test suite:

```bash
python tests/test_rag.py
```

The tests cover three types of queries:

- ✅ **Answerable** — Questions whose answers exist in the knowledge base
- ❌ **Unanswerable** — Questions outside the knowledge base
- ⚠️ **Edge Cases** — Empty user input

### Final Test Results

| ID | Category | Query | Result | Response Time |
|---|---|---|---|---:|
| T01 | Answerable | What is AI and how does it work? | ✅ PASS | 6.02s |
| T02 | Answerable | Explain the concept of machine learning. | ✅ PASS | 11.68s |
| T03 | Unanswerable | What is the meaning of life? | ✅ PASS | 4.60s |
| T04 | Edge Case | Empty query | ✅ PASS | 0.00s |

**Final result: 4/4 tests passed.**

The tests verify that the assistant can:

- Answer questions using information from the documents
- Refuse to invent information when the answer is unavailable
- Handle empty input appropriately
- Provide source information with generated answers

### Performance

Testing showed that retrieval is relatively fast, typically taking around **0.5 seconds**, while local answer generation is the main source of response time.

Several optimizations were evaluated:

- **Top-1 retrieval** was tested, but Top-2 was retained because some answers require information from multiple chunks.
- A **smaller chat model** was tested but did not improve response time.
- The response prompt was made more concise to reduce unnecessary generation.

The final system prioritizes a balance between **answer quality and response time**.

---

## 🚀 Future Improvements

The current system could be extended with:

- 📄 **Additional Document Formats** — Support for PDF and DOCX files
- ✂️ **Advanced Chunking** — More sophisticated document splitting strategies
- 🔎 **Hybrid Search** — Combining semantic and keyword-based search
- 🌍 **Multilingual Support** — Better support for languages other than English
- 💬 **Streaming Responses** — Displaying responses as they are generated
- 🗄️ **Scalable Storage** — Using a dedicated vector database for larger document collections

---

## 📚 References & Learning Resources

This project was developed with reference to the following resources:

- [Microsoft Foundry Local Documentation](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local)
- [Microsoft Learn – Build a RAG Application](https://learn.microsoft.com/en-us/azure/foundry-local/tutorials/tutorial-build-rag-app?tabs=windows)
- [Building Your First Local RAG Application with Foundry Local](https://azurefeeds.com/2026/03/30/building-your-first-local-rag-application-with-foundry-local/)

---

## 🎓 Program

This project was developed as part of the **Microsoft AI Summer Innovators Program**.

---

## License

This project is provided as-is for educational purposes. Modify and distribute as needed.

