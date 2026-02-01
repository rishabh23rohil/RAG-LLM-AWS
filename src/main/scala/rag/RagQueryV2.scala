package rag

import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.search.{IndexSearcher, KnnFloatVectorQuery}
import org.apache.lucene.store.FSDirectory
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._
import scala.util.Using

case class SearchResults(docId: String, chunkId: Int, text: String, score: Float)

object RagQueryV2 {

  private val log = LoggerFactory.getLogger(getClass)

  // Toggle between sequential and MapReduce search
  private val useMapReduce = sys.env.getOrElse("USE_MAPREDUCE_SEARCH", "true").toBoolean

  /**
   * Search a single Lucene shard for top-k results (used by sequential search)
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
   * Search all shards - uses MapReduce if enabled, otherwise sequential.
   */
  def searchAllShards(indexDir: Path, queryVec: Array[Float], k: Int): Seq[SearchResults] = {
    if (useMapReduce) {
      log.info("Using MapReduce for parallel shard search")
      ParallelSearchJob.search(indexDir.toString, queryVec, k)
    } else {
      log.info("Using sequential shard search")
      searchAllShardsSequential(indexDir, queryVec, k)
    }
  }

  /**
   * Sequential search (original implementation).
   */
  private def searchAllShardsSequential(indexDir: Path, queryVec: Array[Float], k: Int): Seq[SearchResults] = {
    val shardDirs = Files.list(indexDir)
      .iterator()
      .asScala
      .filter(p => p.getFileName.toString.startsWith("index_shard_"))
      .toSeq

    if (shardDirs.isEmpty) {
      log.warn(s"No index shards found in $indexDir")
      return Seq.empty
    }

    log.info(s"Searching ${shardDirs.size} shards sequentially...")

    val allResults = shardDirs.flatMap { shardPath =>
      try {
        searchShard(shardPath, queryVec, k)
      } catch {
        case e: Exception =>
          log.error(s"Failed to search shard ${shardPath.getFileName}: ${e.getMessage}")
          Seq.empty[SearchResults]
      }
    }

    allResults.sortBy(-_.score).take(k)
  }

  /**
   * Full RAG pipeline: embed query → search → pack context → generate answer
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
      log.info(s"Embedding query: $query")
      val queryVec = client.embed(Vector(query), model).head

      if (queryVec.isEmpty) {
        return "Error: Failed to generate query embedding"
      }

      log.info(s"Searching index at $indexDir")
      val results = searchAllShards(Paths.get(indexDir), queryVec, topK)

      if (results.isEmpty) {
        return "No relevant documents found in the index."
      }

      log.info(s"Found ${results.size} results")

      val context = results.map { r =>
        s"[${r.docId} - chunk ${r.chunkId}]:\n${r.text}"
      }.mkString("\n\n---\n\n")

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

      val sources = results.map(r => s"${r.docId} (chunk ${r.chunkId})").distinct.mkString(", ")
      s"$answer\n\n[Sources: $sources]"

    } finally {
      client.close()
    }
  }
}