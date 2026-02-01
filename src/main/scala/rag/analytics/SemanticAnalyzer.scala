package rag.analytics

import org.slf4j.LoggerFactory
import java.io.{File, PrintWriter}
import scala.io.Source
import scala.util.Try
import rag.embedding.VectorOps

/**
 * Semantic analysis engine for vocabulary embeddings.
 *
 * Capabilities:
 * - Nearest neighbor discovery
 * - Word pair similarity scoring
 * - Vector analogy computations (A:B :: C:?)
 * - Vocabulary export with full embeddings
 */
object SemanticAnalyzer {

  private val logger = LoggerFactory.getLogger(classOf[SemanticAnalyzer.type])

  case class VocabularyEntry(
    term: String,
    vector: Array[Float],
    magnitude: Float,
    occurrences: Int
  )

  /**
   * Parses embedding output lines.
   * Expected format: "term \t frequency,[v1,v2,...]"
   */
  def parseVocabularyLine(line: String): Option[(String, Array[Float], Int)] = {
    Try {
      if (line == null || line.isBlank) return None

      val vectorStart = line.indexOf('[')
      val vectorEnd = line.lastIndexOf(']')
      if (vectorStart < 0 || vectorEnd < 0) return None

      val prefix = line.substring(0, vectorStart).trim
      val vectorData = line.substring(vectorStart + 1, vectorEnd)
      val vector = vectorData.split(",").map(_.trim.toFloat)

      val parts = prefix.split("[,\\s\\t]+").map(_.trim).filter(_.nonEmpty)
      if (parts.isEmpty) return None

      val term = parts.head
      val frequency = parts.find(_.forall(_.isDigit)).map(_.toInt).getOrElse(1)

      Some((term, vector, frequency))
    }.toOption.flatten
  }

  /**
   * Computes cosine similarity between two vocabulary entries.
   */
  def similarity(entry1: VocabularyEntry, entry2: VocabularyEntry): Double = {
    if (entry1.magnitude == 0 || entry2.magnitude == 0) return 0.0
    if (entry1.vector.length != entry2.vector.length) {
      logger.warn(s"Dimension mismatch: ${entry1.term} vs ${entry2.term}")
      return 0.0
    }
    
    val dotProduct = entry1.vector.zip(entry2.vector).map { case (a, b) => a * b }.sum
    dotProduct / (entry1.magnitude * entry2.magnitude)
  }

  /**
   * Computes vector magnitude (L2 norm).
   */
  def computeMagnitude(vec: Array[Float]): Float = {
    math.sqrt(vec.map(x => x.toDouble * x).sum).toFloat
  }

  /**
   * Executes semantic analysis pipeline on vocabulary data.
   */
  def analyze(inputPath: String): Unit = {
    logger.info(s"[Analyzer] Starting analysis on: $inputPath")
    println("=" * 70)
    println("Semantic Embedding Analysis")
    println("=" * 70)

    // Load vocabulary files
    val inputLocation = new File(inputPath)
    val sourceFiles: Array[File] = if (inputLocation.isDirectory) {
      inputLocation.listFiles().filter(_.getName.startsWith("part-r-"))
    } else if (inputLocation.isFile) {
      Array(inputLocation)
    } else {
      Array.empty
    }

    if (sourceFiles.isEmpty) {
      println("ERROR: No valid input files found")
      return
    }

    println(s"Processing ${sourceFiles.length} input file(s)")

    // Parse all vocabulary entries
    val rawLines = sourceFiles.flatMap { file =>
      println(s"  Loading: ${file.getName}")
      Source.fromFile(file).getLines().filter(_.trim.nonEmpty).toSeq
    }

    val parsedEntries = rawLines.flatMap(parseVocabularyLine)
    val vocabulary = parsedEntries.map { case (term, vec, freq) => 
      term -> (vec, freq) 
    }.toMap

    println(s"\nVocabulary loaded:")
    println(s"  Raw lines: ${rawLines.length}")
    println(s"  Parsed entries: ${parsedEntries.length}")
    println(s"  Unique terms: ${vocabulary.size}")

    if (vocabulary.isEmpty) {
      println("ERROR: No embeddings parsed successfully")
      return
    }

    // Build normalized vocabulary
    val normalizedVocab = vocabulary.map { case (term, (vec, freq)) =>
      VocabularyEntry(term, vec, computeMagnitude(vec), freq)
    }.toArray

    // Export vocabulary CSV
    println("\n[1/4] Exporting vocabulary.csv...")
    exportVocabulary(normalizedVocab)

    // Compute nearest neighbors
    println("\n[2/4] Computing nearest neighbors...")
    computeNearestNeighbors(normalizedVocab)

    // Compute word similarities
    println("\n[3/4] Computing word similarities...")
    computeSimilarities(vocabulary)

    // Compute analogies
    println("\n[4/4] Computing word analogies...")
    computeAnalogies(vocabulary, normalizedVocab)

    println("\n" + "=" * 70)
    println("Analysis Complete")
    println("=" * 70)
    println("Generated files:")
    println("  - vocabulary.csv")
    println("  - nearest_neighbors.csv")
    println("  - word_similarities.csv")
    println("  - word_analogies.csv")
  }

  private def exportVocabulary(vocab: Array[VocabularyEntry]): Unit = {
    val writer = new PrintWriter("vocabulary.csv")
    writer.println("term,id,frequency,vector")
    
    vocab.sortBy(_.term).zipWithIndex.foreach { case (entry, idx) =>
      val vectorStr = entry.vector.map(v => f"$v%.6f").mkString("[", ",", "]")
      writer.println(s"${entry.term},$idx,${entry.occurrences},$vectorStr")
    }
    writer.close()
    println("  vocabulary.csv written")
  }

  private def computeNearestNeighbors(vocab: Array[VocabularyEntry]): Unit = {
    val writer = new PrintWriter("nearest_neighbors.csv")
    writer.println("token,similar_1,cosine_1,similar_2,cosine_2,similar_3,cosine_3,similar_4,cosine_4,similar_5,cosine_5")

    val startTime = System.currentTimeMillis()
    
    vocab.zipWithIndex.foreach { case (entry, idx) =>
      val neighbors = vocab
        .filter(_.term != entry.term)
        .map(other => (other.term, similarity(entry, other)))
        .sortBy(-_._2)
        .take(5)

      val row = neighbors.flatMap { case (term, score) => 
        Seq(term, f"$score%.6f") 
      }.mkString(",")
      writer.println(s"${entry.term},$row")

      if ((idx + 1) % 200 == 0) {
        val elapsed = (System.currentTimeMillis() - startTime) / 1000
        println(f"  Progress: ${idx + 1}%5d / ${vocab.length} (${elapsed}s)")
      }
    }
    writer.close()
    println("  nearest_neighbors.csv written")
  }

  private def computeSimilarities(vocab: Map[String, (Array[Float], Int)]): Unit = {
    val testPairs = Seq(
      ("unity", "unify"),
      ("sweet", "nice"),
      ("sword", "weapon"),
      ("absolute", "abstract"),
      ("ability", "able"),
      ("academic", "accepted")
    )

    val writer = new PrintWriter("word_similarities.csv")
    writer.println("first_term,second_term,similarity_score")

    testPairs.foreach { case (term1, term2) =>
      (vocab.get(term1), vocab.get(term2)) match {
        case (Some((v1, _)), Some((v2, _))) =>
          val mag1 = computeMagnitude(v1)
          val mag2 = computeMagnitude(v2)
          val sim: Double = if (mag1 > 0 && mag2 > 0) {
            (v1.zip(v2).map { case (a, b) => a * b }.sum / (mag1 * mag2)).toDouble
          } else 0.0
          writer.println(f"$term1,$term2,$sim%.6f")
          println(f"  $term1%-15s <-> $term2%-15s: $sim%.4f")
        case _ =>
          println(s"  Missing: $term1 or $term2")
      }
    }
    writer.close()
    println("  word_similarities.csv written")
  }

  private def computeAnalogies(vocab: Map[String, (Array[Float], Int)], normalizedVocab: Array[VocabularyEntry]): Unit = {
    val analogyTests = Seq(
      ("king", "man", "woman", "female"),
      ("city", "country", "paris", "amsterdam"),
      ("day", "night", "summer", "days"),
      ("love", "hate", "good", "nice")
    )

    val writer = new PrintWriter("word_analogies.csv")
    writer.println("term_x,term_y,term_z,prediction,score_1,pred_2,score_2,pred_3,score_3")

    analogyTests.foreach { case (a, b, c, expected) =>
      (vocab.get(a), vocab.get(b), vocab.get(c)) match {
        case (Some((va, _)), Some((vb, _)), Some((vc, _))) =>
          // target = a - b + c
          val target = va.zip(vb).zip(vc).map { case ((x, y), z) => x - y + z }
          val targetMag = computeMagnitude(target)
          
          val predictions = normalizedVocab
            .filterNot(e => Set(a, b, c).contains(e.term))
            .map { entry =>
              val sim: Double = if (targetMag > 0 && entry.magnitude > 0) {
                (target.zip(entry.vector).map { case (x, y) => x * y }.sum / (targetMag * entry.magnitude)).toDouble
              } else 0.0
              (entry.term, sim)
            }
            .sortBy(pair => -pair._2)
            .take(3)

          val predStr = predictions.flatMap { case (term, score) => 
            Seq(term, f"$score%.6f") 
          }.mkString(",")
          writer.println(s"$a,$b,$c,$predStr")
          println(s"  $a - $b + $c => ${predictions.head._1} (expected: $expected)")
        case _ =>
          println(s"  Missing terms for: $a - $b + $c")
      }
    }
    writer.close()
    println("  word_analogies.csv written")
  }
}
