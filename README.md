# RAG-LLM-AWS

[![Scala](https://img.shields.io/badge/Scala-3.5.1-DC322F?style=flat&logo=scala&logoColor=white)](https://www.scala-lang.org/)
[![Hadoop](https://img.shields.io/badge/Hadoop-3.3.6-66CCFF?style=flat&logo=apache-hadoop&logoColor=white)](https://hadoop.apache.org/)
[![Lucene](https://img.shields.io/badge/Lucene-9.10-green?style=flat)](https://lucene.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A distributed **Retrieval-Augmented Generation (RAG)** system built with Scala, Apache Hadoop MapReduce, Apache Lucene HNSW indexes, and Ollama for embedding generation and LLM-based answer synthesis.

---

## Architecture

```mermaid
flowchart LR
    subgraph Input
        PDFs[PDF Documents]
    end

    subgraph MapReduce["Hadoop MapReduce"]
        M1[Mapper 1]
        M2[Mapper 2]
        M3[Mapper N]
        R1[Reducer 1]
        R2[Reducer K]
    end

    subgraph Indexing
        L1[Lucene Shard 1]
        L2[Lucene Shard K]
    end

    subgraph Query["Query Pipeline"]
        QE[Query Embedding]
        KNN[KNN Search]
        CTX[Context Assembly]
        LLM[LLM Generation]
    end

    PDFs --> M1 & M2 & M3
    M1 & M2 & M3 -->|chunks + vectors| R1 & R2
    R1 --> L1
    R2 --> L2
    
    QE --> KNN
    L1 & L2 --> KNN
    KNN --> CTX --> LLM
```

---

## Features

- **Distributed Indexing** — Parallel PDF processing via MapReduce with configurable mappers/reducers
- **Vector Search** — HNSW-based KNN search using Lucene with cosine, Euclidean, or dot-product similarity
- **Query Pipeline** — End-to-end RAG: embed query → search shards → assemble context → generate answer
- **REST API** — Http4s-based endpoints for querying, searching, and health checks
- **Word Analytics** — Vocabulary statistics, semantic neighbors, word analogies, and similarity analysis
- **Cloud Ready** — Deployable on AWS EMR with S3 storage

---

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Scala | 3.5.1 |
| Build Tool | SBT | 1.11.x |
| Distributed Computing | Apache Hadoop MapReduce | 3.3.6 |
| Vector Index | Apache Lucene HNSW | 9.10.0 |
| Embeddings & LLM | Ollama | latest |
| HTTP Server | Http4s + Cats Effect | 0.23.x |
| JSON | Circe | 0.14.x |
| PDF Extraction | Apache PDFBox | 2.0.31 |

---

## Quick Start

### Prerequisites

- JDK 17+
- SBT 1.9+
- Ollama running locally with models:
  ```bash
  ollama pull mxbai-embed-large
  ollama pull llama3
  ```

### Build

```bash
sbt clean compile
sbt assembly  # Creates fat JAR for deployment
```

### Run Locally

**1. Build RAG Index from PDFs:**

```bash
hadoop jar target/scala-3.5.1/RAG-LLM-AWS-assembly-1.0.0.jar rag.Driver \
  Driver rag /path/to/paths.txt /path/to/output mxbai-embed-large COSINE
```

**2. Generate Word Statistics:**

```bash
hadoop jar target/scala-3.5.1/RAG-LLM-AWS-assembly-1.0.0.jar rag.Driver \
  Driver wordstats /path/to/paths.txt /path/to/output mxbai-embed-large COSINE
```

**3. Post-process Embeddings:**

```bash
java -cp target/scala-3.5.1/RAG-LLM-AWS-assembly-1.0.0.jar rag.Driver \
  Driver postprocess /path/to/wordstats/output dummy dummy
```

**4. Start API Server:**

```bash
export RAG_INDEX_PATH=/path/to/lucene-index
export RAG_API_PORT=8080
sbt "runMain rag.RagApiServer"
```

---

## Project Structure

```
RAG-LLM-AWS/
├── src/
│   ├── main/scala/rag/
│   │   ├── Driver.scala              # Entry point for all pipelines
│   │   ├── RagMapper.scala           # PDF → chunks → embeddings
│   │   ├── RagShardReducer.scala     # Builds Lucene HNSW shards
│   │   ├── RagQuery.scala            # Search + answer generation
│   │   ├── RagApiServer.scala        # REST API endpoints
│   │   ├── Ollama.scala              # LLM client for embeddings/chat
│   │   ├── Chunker.scala             # Text segmentation utility
│   │   ├── Vectors.scala             # L2 normalization
│   │   ├── WordStatsMapper.scala     # Vocabulary extraction
│   │   ├── WordStatsReducer.scala    # Word frequency + embeddings
│   │   └── PostProcessEmbeddings.scala # Semantic analysis
│   └── test/scala/rag/
│       └── ...                        # Unit & integration tests
├── project/
│   ├── build.properties
│   └── plugins.sbt
├── build.sbt
└── README.md
```

---

## API Endpoints

### Query (Full RAG)

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is attention mechanism in neural networks?",
    "topK": 5,
    "embedModel": "mxbai-embed-large",
    "chatModel": "llama3"
  }'
```

### Search Only

```bash
curl "http://localhost:8080/api/v1/search?q=neural+networks&topK=5&model=mxbai-embed-large"
```

### Health Check

```bash
curl http://localhost:8080/api/v1/health
```

---

## AWS EMR Deployment

### 1. Upload Bootstrap Script

Upload the bootstrap script to S3 to configure EMR nodes with Ollama.

### 2. Create EMR Cluster

Configure with:
- Instance type: m5.xlarge or larger
- Bootstrap action pointing to your S3 script

### 3. Submit Jobs

```bash
hadoop jar /home/hadoop/RAG-LLM-AWS-assembly-1.0.0.jar rag.Driver \
  Driver rag s3://your-bucket/paths.txt s3://your-bucket/output mxbai-embed-large COSINE
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | mxbai-embed-large | Ollama embedding model |
| `similarity` | COSINE | Vector similarity (COSINE, EUCLIDEAN, DOT_PRODUCT) |
| `linesPerMap` | 50 | PDFs per mapper task |
| `timeoutMs` | 3600000 | Task timeout in milliseconds |
| `numReducers` | 8 | Number of index shards |

---

## Output Files

After running the word statistics pipeline:

| File | Description |
|------|-------------|
| `vocabulary.csv` | Words with token IDs, frequencies, and 1024-dim embeddings |
| `nearest_neighbors.csv` | Top-5 semantic neighbors for each word |
| `word_similarities.csv` | Cosine similarity for test word pairs |
| `word_analogies.csv` | Analogy results (e.g., king - man + woman ≈ queen) |

---

## Demo Video

📺 **[Watch Demo on YouTube](https://youtube.com/YOUR_VIDEO_LINK)**

---

## Author

**Rishabh Rohil**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/rishabh23rohil)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
