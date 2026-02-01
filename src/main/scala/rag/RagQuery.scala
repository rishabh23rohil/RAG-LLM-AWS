package rag

import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.search.{IndexSearcher, KnnFloatVectorQuery}
import org.apache.lucene.store.FSDirectory
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._
import scala.util.Using

case class SearchResult(docId: String, chunkId: Int, text: String, score: Float)

object RagQuery {

  private val log = LoggerFactory.getLogger(getClass)

  /**
   * Performs KNN vector search on a single index shard.
   */
  def searchShard(shardPath: Path, queryVec: Array[Float], k: Int): Seq[SearchResults] = {
    Using.resource(FSDirectory.open(shardPath)) { directory =>
      Using.resource(DirectoryReader.open(directory)) { reader =>
        val searcher = new IndexSearcher(reader)
        val query = new KnnFloatVectorQuery("vec", queryVec, k)
        val topDocs = searcher.search(query, k)
        val storedFields = searcher.storedFields()

        topDocs.scoreDocs.map { scoreDoc =>
          val doc = storedFields.document(scoreDoc.doc)
          SearchResults(
            docId = doc.get("doc_id"),
            chunkId = doc.get("chunk_id").toInt,
            text = doc.get("text"),
            score = scoreDoc.score
          )
        }.toSeq
      }
    }
  }

  /**
   * Distributed search across all index partitions with result aggregation.
   */
  def searchAllShards(indexDir: Path, queryVec: Array[Float], k: Int): Seq[SearchResults] = {
    val shardDirs = Files.list(indexDir)
      .iterator()
      .asScala
      .filter(p => p.getFileName.toString.startsWith("index_shard_"))
      .toSeq

    if (shardDirs.isEmpty) {
      log.warn(s"No index shards found in $indexDir")
      return Seq.empty
    }

    log.info(s"Searching ${shardDirs.size} shards in parallel...")

    // Fan-out: search each shard in parallel
    // Use .toVector.par or just sequential processing
    val allResults = shardDirs.flatMap { shardPath =>
      try {
        searchShard(shardPath, queryVec, k)
      } catch {
        case e: Exception =>
          log.error(s"Failed to search shard ${shardPath.getFileName}: ${e.getMessage}")
          Seq.empty[SearchResults]
      }
    }

    // Fan-in: merge results using global top-k (min-heap approach)
    allResults.sortBy(-_.score).take(k)
  }

  /**
   * Complete RAG execution: query embedding → vector search → context assembly → LLM generation
   */
  def answer(
              query: String,
              indexDir: String,
              model: String = "mxbai-embed-large",
              topK: Int = 5,
              chatModel: String = "llama3"
            ): String = {

    val client = new Ollama()

    try {
      // Step 1: Embed the query
      log.info(s"Embedding query: $query")
      val queryVec = client.embed(Vector(query), model).head

      if (queryVec.isEmpty) {
        return "Error: Failed to generate query embedding"
      }

      // Step 2: Search Lucene index (fan-out/fan-in across shards)
      log.info(s"Searching index at $indexDir")
      val results = searchAllShards(Paths.get(indexDir), queryVec, topK)

      if (results.isEmpty) {
        return "No relevant documents found in the index."
      }

      log.info(s"Found ${results.size} results")

      // Step 3: Pack context from top results
      val context = results.map { r =>
        s"[${r.docId} - chunk ${r.chunkId}]:\n${r.text}"
      }.mkString("\n\n---\n\n")

      // Step 4: Generate answer using Ollama /api/chat
      log.info("Generating answer with LLM...")
      val systemMessage = ChatMessage(
        role = "system",
        content = "You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the answer cannot be found in the context, explicitly state that."
      )

      val userMessage = ChatMessage(
        role = "user",
        content = s"""Context from documents:
$context

Question: $query

Provide a clear, concise answer based only on the context above."""
      )


      val answer = client.chat(Vector(systemMessage, userMessage), chatModel)

      // Include source attribution
      val sources = results.map(r => s"${r.docId} (chunk ${r.chunkId})").distinct.mkString(", ")
      s"$answer\n\n[Sources: $sources]"

    } finally {
      client.close()
    }
  }
}