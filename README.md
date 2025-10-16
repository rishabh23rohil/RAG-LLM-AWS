# CS441 HW1 — RAG Index Builder on Hadoop/Lucene/Ollama

**Author:** Rishabh Rohil  
**Email:** <your_uic_email@uic.edu>  

## Overview
- Build Lucene HNSW index over MSR PDFs with Ollama embeddings
- Query via kNN (`KnnSearch`)
- Produce vocabulary frequency + similarity/analogy CSVs

## Prereqs
- JDK 17+, SBT 1.9+
- Ollama running and `OLLAMA_HOST` set
- (Option 1 path) Hadoop local & EMR for cloud run

## Quick Start (Local)
```bash
sbt clean compile
sbt "runMain rag.mr.JobRunner file:///.../local_paths.txt file:///.../var/mrstatus_XXX"
# copy shard dirs -> ./lucene-index/shard-*
sbt "runMain rag.lucene.InnnSearch lucene-index neural networks and attention mechanisms 5 default"
sbt "runMain rag.stats.VocabStats"
sbt "runMain rag.stats.SimilarityEval"
