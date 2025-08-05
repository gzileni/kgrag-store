# Ingestion


**KGragRetriever** è una classe che esegue il download, process e gestisce documenti in formato PDF, CSV o JSON sia da filesystem locale che da AWS S3.

## Parametri Principali

| Parametro               | Tipo                       | Descrizione                                     |
| ----------------------- | -------------------------- | ----------------------------------------------- |
| `format_file`           | `"pdf" \| "csv" \| "json"` | Formato del file da processare (default: "pdf") |
| `path_type`             | `"fs" \| "s3"`             | Origine dei dati: locale (`fs`) o S3 (`s3`)     |
| `path`                  | `str`                      | Percorso locale (richiesto se `path_type=fs`)   |
| `aws_access_key_id`     | `str`                      | Chiave AWS (richiesto se `path_type=s3`)        |
| `aws_secret_access_key` | `str`                      | Secret AWS (richiesto se `path_type=s3`)        |
| `s3_bucket`             | `str`                      | Nome bucket S3 (richiesto se `path_type=s3`)    |
| `aws_region`            | `str`                      | Regione AWS S3 (richiesto se `path_type=s3`)    |
| `path_download`         | `str`                      | Directory locale per scaricare i file da S3     |
| `thread`                | `str`                      | Trace ID per il logging                         |

###


### 1. Caricamento da filesystem locale

Supponiamo di avere un PDF in `/tmp/test.pdf`:

```python
from qi_memory import KGragRetriever

retriever = KGragRetriever(
    format_file="pdf",
    path_type="fs",
    path="/tmp/test.pdf"
)

documents = retriever.process()
print(documents)  # Restituisce una lista di oggetti Document
```

**Per CSV:**

```python
retriever = KGragRetriever(
    format_file="csv",
    path_type="fs",
    path="/tmp/dataset.csv"
)
documents = retriever.process()
```

### 2. Caricamento da AWS S3

Supponiamo di voler processare i PDF dal bucket `my-bucket` con prefisso `docs/`:

```python
retriever = KGragRetriever(
    format_file="pdf",
    path_type="s3",
    aws_access_key_id="TUO_ACCESS_KEY",
    aws_secret_access_key="TUO_SECRET_KEY",
    s3_bucket="my-bucket",
    aws_region="eu-west-1",
    path_download="/tmp/downloads"
)

# processa TUTTI i file PDF sotto il prefisso 'docs/'
documents = retriever.process(prefix="docs/", limit=10) # limita a 10 file, opzionale
```

> I file vengono scaricati in `/tmp/downloads` e poi processati.

### Gestione Errori

* Se manca un parametro richiesto (es: path o credenziali S3), viene sollevato `ValueError` con logging automatico.
* Se il formato non è supportato (`format_file` ≠ `"pdf"`, `"csv"`, `"json"`), viene sollevato errore.
* La directory di download viene creata e pulita automaticamente a ogni esecuzione da S3 (`delete=True` di default).

### Esempio avanzato con parametri dinamici

```python
# Parametri variabili a runtime:
config = {
    "format_file": "json",
    "path_type": "fs",
    "path": "/tmp/file.json"
}
retriever = KGragRetriever(**config)
documents = retriever.process()
```

## Qdrant + Neo4j GraphRAG

**MemoryStoreGraph** permette di avere un RAG avanzato con Vector Store e Knowledge Graph. 
Combina la ricerca semantica di **Qdrant** (vector DB) con la potenza relazionale di **Neo4j** (knowledge graph) in un’architettura **GraphRAG**.
Questo permette retrieval avanzato (semantic + graph), arricchendo la generazione delle risposte LLM con contesto e relazioni complesse tra entità.

### Vantaggi principali

* **Recall & Precision migliorati**: Qdrant trova risultati semanticamente rilevanti; Neo4j garantisce coerenza tramite relazioni e grafo.
* **Comprensione contestuale**: Neo4j modella le relazioni tra entità, migliorando la precisione della risposta LLM.
* **Adattabilità a query complesse**: Puoi gestire domande multi-hop, ragionamento e contesti ramificati grazie al grafo.
* **Scalabilità e costi**: Qdrant offre vector search veloce ed economica. Neo4j aggiunge la profondità senza caricare tutto su LLM.
  È possibile anche adottare NER-based extraction per ottimizzare i costi.

### Configurazione (variabili ambiente consigliate)

| Variabile        | Descrizione                           |
| ---------------- | ------------------------------------- |
| `NEO4J_URL`      | URL Neo4j (es: bolt://localhost:7687) |
| `NEO4J_USERNAME` | Username Neo4j                        |
| `NEO4J_PASSWORD` | Password Neo4j                        |
| `OPENAI_API_KEY` | (opzionale) API key OpenAI            |

Puoi anche passare questi parametri direttamente al costruttore.

#### 1. Inizializzazione

```python
from memory_graph_module import MemoryStoreGraph

graph_store = MemoryStoreGraph(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="your_password",
    llm_type="openai",  # oppure "ollama"
    llm_model="gpt-4o-2024-08-06",  # nome modello LLM
    llm_model_url=None  # se usi Ollama/vLLM
)
```

#### 2. Ingestion

```python
import asyncio

# Supponiamo di avere una lista di oggetti Document di LangChain
documents = [Document(page_content="Alan Turing was a mathematician who worked with John von Neumann.")]
await graph_store.ingestion_batch(documents)
```

```python
import os
import asyncio
from dotenv import load_dotenv
from qi_memory import MemoryStoreGraph, KGragRetriever

load_dotenv()

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
s3_bucket = os.getenv('AWS_BUCKET_NAME')
aws_region = os.getenv('AWS_REGION')

app_root = os.path.dirname(os.path.abspath(__file__))
path_download = os.path.join(app_root, ".tmp")

retriever = KGragRetriever(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    s3_bucket=s3_bucket,
    aws_region=aws_region,
    path_type="s3",
    path_download=path_download,
    format_file="pdf"
)

documents = retriever.process(limit=3, prefix="rt")
print(f"Retrieved {len(documents)} documents.")

graph = MemoryStoreGraph(llm_model=os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'))

async def main():
    await graph.ingestion_batch(documents=documents, collection_name="test_collection")

asyncio.run(main())
```

#### 3. Query

```python
result = await graph_store.query("Who collaborated with Alan Turing?")
print(result)
```

### Architettura: come funziona

1. **Ingestion**:

   * Estrae nodi e relazioni dal testo (LLM-based NER/IE)
   * Salva nodi e archi in Neo4j
   * Estrae embedding testuali e salva in Qdrant

2. **Query**:

   * Trasforma la domanda in embedding
   * Recupera i vettori più rilevanti da Qdrant
   * Trova i nodi e relazioni correlate in Neo4j (subgraph extraction)
   * Costruisce un contesto testuale dal grafo
   * Invia il contesto + domanda a LLM per generare la risposta

### Parametri principali

| Parametro        | Tipo            | Descrizione                                              |
| ---------------- | --------------- | -------------------------------------------------------- |
| `neo4j_uri`      | `str`           | Indirizzo del DB Neo4j                                   |
| `neo4j_username` | `str`           | Username Neo4j                                           |
| `neo4j_password` | `str`           | Password Neo4j                                           |
| `llm_type`       | `"openai"`, ... | Motore LLM (`"openai"`, `"ollama"`, `"vllm"`)            |
| `llm_model`      | `str`           | Nome modello LLM (es: `"gpt-4o-2024-08-06"`, `"llama3"`) |
| `llm_model_url`  | `str`           | (Se serve) endpoint Ollama/vLLM                          |

### Funzioni e metodi chiave

* `ingestion_batch(documents, collection_name=None)`
  Ingesta una lista di documenti LangChain
* `query(query_str, collection_name=None)`
  Recupera risposte semantiche+relazionali dal grafo

### Best practices e note

* Prevedi una **pulizia regolare** delle collezioni e subgraph in fase di test.
* In produzione, usa Neo4j e Qdrant con autenticazione e connessioni sicure.
* Sfrutta l’integrazione con Named Entity Recognition per estrazione efficiente in batch massivi.

![graph](./graph.png)

### Link utili

* [Qdrant GraphRAG Example](https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/)
* [Neo4j Python Driver](https://neo4j.com/docs/api/python-driver/current/)
* [LangChain Document Loader](https://python.langchain.com/docs/modules/data_connection/document_loaders/community/)

## Aggiornamento e Deploy della Libreria

Per aggiornare la libreria ed eseguire il deploy:

1. Clona il progetto da:

    ```bash
    git clone https://gzileni_questit@bitbucket.org/questit-dev/qi_memory.git
    ```

2. Aggiorna la versione nel file `pyproject.toml`.
3. Esegui lo script di deploy:

    ```bash
    ./deploy.sh
    ```

## Installazione

```python
pip install --upgrade --trusted-host 3.71.130.76 --index-url http://agent:agent@3.71.130.76/simple qi-memory
```
