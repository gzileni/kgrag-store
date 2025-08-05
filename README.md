# krag-store

## 📦 Installation

Install the `krag-store` library via pip:

```bash
pip install krag-store
```                

## Ingestion Data

**KGragRetriever** is a class for **ingesting** and **processing** documents in PDF, CSV, or JSON format, coming from either the **local filesystem** or **AWS S3**.
It natively integrates **Qdrant** (Vector Store) and **Neo4j** (Knowledge Graph) for semantic and relational retrieval (GraphRAG).

* **Centralized configuration**: all parameters are automatically loaded from the [`settings`](./kgrag_store/kgrag_config.py) module, so you don't need to pass them manually to the constructor.
* **Dynamic support for local and cloud LLMs**:

    * OpenAI, Azure OpenAI
    * Ollama
    * vLLM
* **Automatic loading of `.env` files** based on the environment (`development`, `staging`, `production`, `test`).
* **LLM and embedding endpoints are generated automatically** for Ollama/vLLM.

---

### 📂 `.env` File Structure

Example for development environment:

```env
APP_ENV=development

# LLM settings
LLM_MODEL_TYPE=openai
LLM_MODEL_NAME=gpt-4.1-mini
MODEL_EMBEDDING=text-embedding-3-small

# Neo4j
NEO4J_URL=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Qdrant
QDRANT_URL=http://localhost:6333

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=4
```

To use **Ollama** locally:

```env
LLM_MODEL_TYPE=ollama
LLM_URL=http://localhost:11434
LLM_MODEL_NAME=llama3
MODEL_EMBEDDING=all-minilm
```

---

### ⚙️ Main Parameters of `KGragRetriever`

With the new management, parameters are already set in `settings` and **do not need to be passed manually**.
The constructor receives values from:

```python
from kgrag_store import settings
```

Current example:

```python
from kgrag_store import settings, KGragRetriever

model_embedding_url = None
llm_model_url = None

if settings.LLM_MODEL_TYPE == "ollama" and settings.LLM_URL:
        model_embedding_url = f"{settings.LLM_URL}/api/embeddings"
        llm_model_url = settings.LLM_URL

kgrag = KGragRetriever(
        path_type="fs",
        path_download=settings.PATH_DOWNLOAD,
        format_file="pdf",
        collection_name=settings.COLLECTION_NAME,
        llm_model=settings.LLM_MODEL_NAME,
        llm_type=settings.LLM_MODEL_TYPE,
        llm_model_url=llm_model_url,
        model_embedding_type=settings.LLM_MODEL_TYPE,
        model_embedding_name=settings.MODEL_EMBEDDING,
        model_embedding_url=model_embedding_url,
        model_embedding_vs_name=settings.VECTORDB_SENTENCE_MODEL,
        model_embedding_vs_type=settings.VECTORDB_SENTENCE_TYPE,
        model_embedding_vs_path=settings.VECTORDB_SENTENCE_PATH,
        neo4j_url=settings.NEO4J_URL,
        neo4j_username=settings.NEO4J_USERNAME,
        neo4j_password=settings.NEO4J_PASSWORD,
        neo4j_db_name=settings.NEO4J_DB_NAME,
        qdrant_url=settings.QDRANT_URL,
        redis_host=settings.REDIS_HOST,
        redis_port=settings.REDIS_PORT,
        redis_db=settings.REDIS_DB
)
```

---

### 📥 Usage Examples

#### 1️⃣ Ingestion from Local Filesystem

```python
documents = kgrag.process_documents(path="./data")
print(f"Processed {len(documents)} documents.")
```

This example shows how to load and process PDF documents using the `langchain_community` library.  
The code uses `PyPDFLoader` to read a PDF file from the `./data/` directory, loads the documents, and processes them with the `kgrag.process_documents` function.  
Finally, it prints the number of processed documents.

```python
from langchain_community.document_loaders import PyPDFLoader

path = "./data/my_pdf.pdf"
loader = PyPDFLoader(path)
documents = loader.load()
documents = kgrag.process_documents(documents=documents)
print(f"Processed {len(documents)} documents.")
```

---

#### 2️⃣ Ingestion from AWS S3

Just set in your `.env`:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_BUCKET_NAME=my-bucket
AWS_REGION=eu-west-1
```

Then:

```python
kgrag.path_type = "s3"
documents = kgrag.process_documents(
    prefix="docs/", 
    limit=10,
    bucket_name=os.getenv("AWS_BUCKET_NAME),
    aws_region=os.getenv("AWS_REGION)
)
```

---

### 🧠 GraphRAG with Qdrant + Neo4j

Retrieval is performed by combining:

* **Qdrant** → semantic search on embeddings
* **Neo4j** → extraction of relational subgraphs

Advantages:

* Improved **recall** and **precision**
* Advanced contextual understanding
* Support for multi-hop queries

### 📜 Architecture

1. **Ingestion**:

     * Parsing and normalization of documents
     * Embedding extraction → Qdrant
     * Entity and relation extraction → Neo4j


## Relations and Nodes

Perfetto 👍
Visto il codice che mi hai fornito, ti preparo un **README.md** chiaro e strutturato che spieghi **KGragGraph**, il ruolo di **extract\_graph\_components**, e le variabili di ambiente necessarie.
Ecco la versione pronta:

---

## 🧠 KGragGraph

`KGragGraph` is an advanced extension of **KGragVectorStore** that implements a **GraphRAG** architecture:
it combines **Qdrant** (Vector Store) and **Neo4j** (Knowledge Graph) for extraction, ingestion, and querying of structured knowledge.

### 🚀 Main Features

* **Graph component extraction with LLM**
    Using the `extract_graph_components` method, the system analyzes text to identify:

    * **Nodes** (unique entities)
    * **Relationships** (edges between nodes)
* **Automatic ingestion into Neo4j**
    Creation of nodes and edges in a knowledge graph.
* **Vectorization and storage in Qdrant**
    Each document is transformed into embeddings for semantic search.
* **Support for multiple LLMs**:

    * `openai` (ChatGPT / GPT-4.x)
    * `ollama` (local LLaMA models)
    * `vllm` (accelerated LLM server)
* **Advanced queries**:

    * Subgraph retrieval
    * Context composition
    * LLM-generated answers with graph knowledge

---

## ⚙️ Required Environment Variables

| Variable                          | Default                 | Description                                                                    |
| ---------------------------------- | ----------------------- | ------------------------------------------------------------------------------ |
| **Neo4j**                         |                         |                                                                                |
| `NEO4J_URL`                       | *(required)*            | Neo4j connection URI (e.g. `bolt://localhost:7687` or `neo4j://host:port`)     |
| `NEO4J_USERNAME`                  | *(required)*            | Username for Neo4j                                                             |
| `NEO4J_PASSWORD`                  | *(required)*            | Password for Neo4j                                                             |
| `NEO4J_DB_NAME`                   | *(optional)*            | Database name (if different from default)                                      |
| **LLM**                           |                         |                                                                                |
| `LLM_TYPE`                        | `openai`                | Model type: `openai`, `ollama`, `vllm`                                         |
| `LLM_MODEL`                       | `gpt-4o-2024-08-06`     | LLM model name                                                                 |
| `LLM_URL`                         | *(optional)*            | Model endpoint (required for `ollama`/`vllm`)                                  |
| **Qdrant**                        |                         |                                                                                |
| `QDRANT_URL`                      | `http://localhost:6333` | Qdrant endpoint                                                                |
| **OpenAI** (if `LLM_TYPE=openai`) |                         |                                                                                |
| `OPENAI_API_KEY`                  | *(required)*            | OpenAI API key                                                                 |

---

### 📜 Key Method: `extract_graph_components`

This is the core of **KGragGraph**:
it extracts entities and relationships from unstructured text using an **LLM**.

#### Method Flow

1. **Prompt construction**
     The input text is inserted into a template that asks the LLM to extract nodes and relationships.
2. **Response parsing**
     The LLM returns a JSON that is validated as `GraphComponents`.
3. **Dictionary creation**:

     * `nodes` → dictionary `{node_name: uuid}`
     * `relationships` → list of relationships `{source, target, type}`
4. **Output**:

     ```python
     (
             {"Alan Turing": "uuid-123", "John von Neumann": "uuid-456"},
             [{"source": "uuid-123", "target": "uuid-456", "type": "collaborated_with"}]
     )
     ```

---

### 🔄 Ingestion

Ingestion is divided into 3 phases:

```python
nodes, relationships = await graph.extract_graph_components(raw_text)
graph.ingest_to_neo4j(nodes, relationships)

# Optional
await graph.ingest_to_qdrant(raw_data=raw_text, node_id_mapping=nodes)
```

**Note:** The `ingest_to_qdrant` function can be replaced with any other vector database, or you can continue using Qdrant depending on your project requirements.

---

#### 📝 Custom Prompt Example

You can provide an additional `prompt_user` parameter to reinforce or customize the base prompt used for entity and relationship extraction. This is useful when you want to guide the LLM to focus on specific types of entities or relationships.

```python
custom_instruction = (
    "Extract only PERSON and ORGANIZATION entities, and identify 'works_for' relationships."
)

nodes, relationships = await graph.extract_graph_components(
    raw_text,
    prompt_user=custom_instruction
)
```

This will instruct the LLM to parse the text according to your custom requirements, improving extraction accuracy for your use case.

---

### 💡 Usage Example

```python
from kgrag_graph import KGragGraph
from langchain_core.documents import Document

graph = KGragGraph(
        neo4j_url="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        llm_type="openai",
        llm_model="gpt-4o-2024-08-06"
)

# Example document
doc = Document(page_content="Alan Turing collaborated with John von Neumann in computer science.")

# Ingestion
async for status in graph._ingestion_batch([doc]):
        print(status)
```

## 🔍 Graph Querying

The **KGragGraph** class offers two modes for querying the graph:

### 1️⃣ `query_stream`

Executes the query in **asynchronous streaming** mode, returning partial responses as the LLM generates them.

```python
async for partial_answer in graph.query_stream("Who collaborated with Alan Turing?"):
    print(partial_answer)
```

**Parameters**

| Name         | Type        | Default          | Description                                                      |
| ------------ | ----------- | ---------------- | ---------------------------------------------------------------- |
| `query`      | `str`       | *(required)*     | The question to ask the graph                                    |
| `entity_ids` | `list[Any]` | `None`           | List of entity IDs to limit the search scope (optional)          |

**How it works**

1. Retrieves the graph context with `_get_graph_context`:
   * If `entity_ids` is not provided, it infers them with `retrieve_ids(query)`
   * Extracts related subgraphs from Neo4j
2. Generates the answer with `_stream`, producing a continuous partial output.

**When to use**

* To display the answer in real time.
* For interactive use cases (chatbots, reactive UIs).

---

### 2️⃣ `query`

Executes the query in **standard asynchronous** mode, returning the complete answer only after processing is finished.

```python
answer = await graph.query("Who collaborated with Alan Turing?")
print(answer)
```

**Parameters**

| Name         | Type        | Default          | Description                                                      |
| ------------ | ----------- | ---------------- | ---------------------------------------------------------------- |
| `query`      | `str`       | *(required)*     | The question to ask the graph                                    |
| `entity_ids` | `list[Any]` | `None`           | List of entity IDs to limit the search scope (optional)          |

**How it works**

1. Retrieves the graph context with `_get_graph_context`
2. Runs `_run` to obtain the final answer as a single string

**When to use**

* When you need the complete answer at once.
* For batch analysis or non-interactive logic.

---

**Main differences** between `query_stream` and `query`:

| Feature           | `query_stream`                | `query`                              |
| ----------------- | ---------------------------- | ------------------------------------ |
| Output mode       | Streaming                     | Complete                             |
| Perceived speed   | High (immediate results)      | Depends on total processing time     |
| Typical use       | Interactive UIs, chatbots     | Batch analysis, reporting            |

