package rag.search

import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.search.{IndexSearcher, KnnFloatVectorQuery}
import org.apache.lucene.store.FSDirectory
import org.slf4j.LoggerFactory
import rag.embedding.{OllamaClient, ConversationMessage}

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._
import scala.util.Using

/**
 * Result from vector similarity search.
 */
case class QueryResult(
  documentKey: String,
  segmentIndex: Int,
  content: String,
  relevanceScore: Float
)

/**
 * Vector search engine for querying distributed HNSW indexes.
 *
 * Supports:
 * - Single partition search
 * - Fan-out/fan-in across partitions
 * - End-to-end RAG with answer generation
 */
object VectorSearchEngine {

  private val logger = LoggerFactory.getLogger(classOf[VectorSearchEngine.type])

  /**
   * Searches a single index partition for nearest neighbors.
   */
  def searchPartition(partitionPath: Path, queryVector: Array[Float], limit: Int): Seq[QueryResult] = {
    Using.resource(FSDirectory.open(partitionPath)) { directory =>
      Using.resource(DirectoryReader.open(directory)) { reader =>
        val searcher = new IndexSearcher(reader)
        val knnQuery = new KnnFloatVectorQuery("vector", queryVector, limit)
        val hits = searcher.search(knnQuery, limit)
        val fieldReader = searcher.storedFields()

        hits.scoreDocs.map { hit =>
          val doc = fieldReader.document(hit.doc)
          QueryResult(
            documentKey = doc.get("document_key"),
            segmentIndex = doc.get("segment_index").toInt,
            content = doc.get("content"),
            relevanceScore = hit.score
          )
        }.toSeq
      }
    }
  }

  /**
   * Searches all partitions and merges results.
   */
  def searchAllPartitions(indexRoot: Path, queryVector: Array[Float], limit: Int): Seq[QueryResult] = {
    val partitions = Files.list(indexRoot)
      .iterator()
      .asScala
      .filter(p => p.getFileName.toString.startsWith("partition_"))
      .toSeq

    if (partitions.isEmpty) {
      logger.warn(s"[Search] No partitions found in $indexRoot")
      return Seq.empty
    }

    logger.debug(s"[Search] Querying ${partitions.size} partitions")

    val aggregatedResults = partitions.flatMap { partition =>
      try {
        searchPartition(partition, queryVector, limit)
      } catch {
        case ex: Exception =>
          logger.error(s"[Search] Partition ${partition.getFileName} failed: ${ex.getMessage}")
          Seq.empty
      }
    }

    // Merge and select top-k globally
    aggregatedResults.sortBy(-_.relevanceScore).take(limit)
  }

  /**
   * Full RAG pipeline: embed → search → context → generate.
   */
  def answerQuery(
    question: String,
    indexPath: String,
    embeddingModel: String = "mxbai-embed-large",
    resultLimit: Int = 5,
    generationModel: String = "llama3"
  ): String = {
    val llmClient = new OllamaClient()

    try {
      // Generate query embedding
      logger.debug(s"[RAG] Embedding query: ${question.take(50)}...")
      val queryEmbedding = llmClient.generateEmbeddings(Vector(question), embeddingModel).head

      if (queryEmbedding.isEmpty) {
        return "Error: Could not generate embedding for query"
      }

      // Search index partitions
      val searchResults = searchAllPartitions(Paths.get(indexPath), queryEmbedding, resultLimit)

      if (searchResults.isEmpty) {
        return "No relevant content found in the knowledge base."
      }

      logger.debug(s"[RAG] Retrieved ${searchResults.size} relevant segments")

      // Build context from retrieved segments
      val contextText = searchResults.map { result =>
        s"[Source: ${result.documentKey}, Segment ${result.segmentIndex}]\n${result.content}"
      }.mkString("\n\n---\n\n")

      // Generate answer using LLM
      val systemPrompt = ConversationMessage(
        role = "system",
        content = "You are a knowledgeable assistant. Answer questions using ONLY the provided context. If the context doesn't contain sufficient information, state that clearly."
      )

      val userPrompt = ConversationMessage(
        role = "user",
        content = s"""Reference Material:
$contextText

User Question: $question

Provide a comprehensive answer based solely on the reference material above."""
      )

      val generatedAnswer = llmClient.generateCompletion(Vector(systemPrompt, userPrompt), generationModel)

      // Format response with citations
      val citations = searchResults.map(r => s"${r.documentKey}:${r.segmentIndex}").distinct.mkString(", ")
      s"$generatedAnswer\n\n[References: $citations]"

    } finally {
      llmClient.shutdown()
    }
  }
}
