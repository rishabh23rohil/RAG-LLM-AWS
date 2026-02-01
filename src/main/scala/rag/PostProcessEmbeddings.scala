package rag

import org.slf4j.LoggerFactory
import java.io.{File, PrintWriter}
import scala.io.Source

/**
 * Semantic analysis utilities for word embeddings.
 *
 * Features:
 * - Functional implementation (immutable data structures)
 * - Parses vocabulary with frequency counts and vectors
 * - Exports complete 1024-dimensional embeddings
 * - Computes nearest neighbors, word similarities, and analogies
 */
object PostProcessEmbeddings {

  private val log = LoggerFactory.getLogger(getClass)

  case class NormalizedWord(word: String, vec: Array[Float], norm: Float, frequency: Int)

  def cosineSimilarity(v1: Array[Float], norm1: Float, v2: Array[Float], norm2: Float): Double =
    if (norm1 == 0 || norm2 == 0) 0.0
    else if (v1.length != v2.length) {
      log.error(s"Vector dimension mismatch: ${v1.length} vs ${v2.length}")
      0.0
    } else {
      val dot = v1.zip(v2).foldLeft(0.0f) { case (sum, (a, b)) => sum + a * b }
      dot / (norm1 * norm2)
    }

  def computeNorm(v: Array[Float]): Float =
    math.sqrt(v.foldLeft(0.0f)((sum, x) => sum + x * x)).toFloat

  def add(v1: Array[Float], v2: Array[Float]): Array[Float] =
    v1.zip(v2).map { case (x, y) => x + y }

  def sub(v1: Array[Float], v2: Array[Float]): Array[Float] =
    v1.zip(v2).map { case (x, y) => x - y }

  /**
   * Parse lines like:
   * "word word,<freq>,[v1,v2,...]"
   * "word,<freq>,[v1,v2,...]"
   */
  def parseLine(line: String): Option[(String, Array[Float], Int)] = {
    try {
      if (line == null || line.trim.isEmpty) return None

      val bracketStart = line.indexOf('[')
      val bracketEnd = line.lastIndexOf(']')
      if (bracketStart == -1 || bracketEnd == -1) return None

      val beforeVec = line.substring(0, bracketStart).trim
      val vecStr = line.substring(bracketStart + 1, bracketEnd)
      val vec = vecStr.split(",").map(_.trim.toFloat)

      val tokens = beforeVec.split("[,\\s]+").map(_.trim).filter(_.nonEmpty)
      if (tokens.isEmpty) return None

      val word = tokens.head
      val freqToken = tokens.find(_.forall(_.isDigit)).getOrElse("1")
      val frequency = freqToken.toInt

      Some((word, vec, frequency))
    } catch {
      case e: Exception =>
        log.warn(s"Failed to parse line: ${line.take(100)}... - ${e.getMessage}")
        None
    }
  }

  def run(inputPath: String): Unit = {
    println("=" * 80)
    println("Starting Word Embedding Post-Processing (IMMUTABLE VERSION)")
    println("=" * 80)
    println(s"Input path: $inputPath")

    // Discover input files
    val inputFile = new File(inputPath)
    val filesToProcess: Array[File] =
      if (inputFile.isDirectory)
        inputFile.listFiles().filter(_.getName.startsWith("part-r-"))
      else if (inputFile.isFile)
        Array(inputFile)
      else
        Array.empty[File]

    if (filesToProcess.isEmpty) {
      println(" No valid input files found.")
      System.exit(1)
    }

    println(s"Found ${filesToProcess.length} file(s) to process")
    println("Parsing vocabulary...")

    val allLines = filesToProcess.flatMap { file =>
      println(s"  Reading file: ${file.getName}")
      Source.fromFile(file).getLines().filter(_.trim.nonEmpty).toSeq
    }

    val parsed = allLines.flatMap(parseLine)
    val vocab = parsed.map { case (word, vec, freq) => word -> (vec, freq) }.toMap

    println(s"\nParsing complete:")
    println(s"  Total lines read: ${allLines.length}")
    println(s"  Successfully parsed: ${parsed.length}")
    println(s"  Vocabulary size: ${vocab.size}")

    if (vocab.isEmpty) {
      println(" No valid embeddings loaded.")
      System.exit(1)
    }

    val normalizedVocab = vocab.map { case (word, (vec, freq)) =>
      NormalizedWord(word, vec, computeNorm(vec), freq)
    }.toArray

    println(s"\nPre-computed norms for ${normalizedVocab.length} words")

    // Export vocabulary
    println("\nExporting vocabulary.csv (full 1024-d embeddings)...")
    val vocabWriter = new PrintWriter("vocabulary.csv")
    vocabWriter.println("word,token_id,frequency,embedding")
    normalizedVocab.sortBy(_.word).zipWithIndex.foreach { case (nw, idx) =>
      val vecString = nw.vec.map(v => f"$v%.6f").mkString("[", ",", "]")
      vocabWriter.println(s"${nw.word},$idx,${nw.frequency},$vecString")
    }
    vocabWriter.close()
    println("✓ vocabulary.csv written successfully")

    // Compute nearest neighbors
    println("\n" + "=" * 80)
    println("Task 1: Nearest Neighbors")
    println("=" * 80)
    val pwNN = new PrintWriter("nearest_neighbors.csv")
    pwNN.println("word,neighbor_1,sim_1,neighbor_2,sim_2,neighbor_3,sim_3,neighbor_4,sim_4,neighbor_5,sim_5")

    val startTime = System.currentTimeMillis()

    normalizedVocab.zipWithIndex.foreach { case (nw, idx) =>
      val similarities = normalizedVocab
        .filter(_.word != nw.word)
        .map(other => (other.word, cosineSimilarity(nw.vec, nw.norm, other.vec, other.norm)))
        .sortBy(-_._2)
        .take(5)

      val row = similarities.flatMap { case (w, s) => Seq(w, f"$s%.6f") }.mkString(",")
      pwNN.println(s"${nw.word},$row")

      if ((idx + 1) % 100 == 0) {
        val elapsed = (System.currentTimeMillis() - startTime) / 1000
        println(f"  Progress: ${idx + 1}%5d / ${normalizedVocab.length}%5d (${elapsed}s elapsed)")
      }
    }
    pwNN.close()
    println("✓ nearest_neighbors.csv written")

    // Word similarities
    println("\n" + "=" * 80)
    println("Task 2: Word Similarities")
    println("=" * 80)
    val testPairs = Seq(
      ("unity", "unify"),
      ("sweet", "nice"),
      ("sword", "weapon"),
      ("absolute", "abstract"),
      ("ability", "able"),
      ("academic", "accepted")
    )

    val simWriter = new PrintWriter("word_similarities.csv")
    simWriter.println("word_1,word_2,cosine_similarity")

    testPairs.foreach { case (w1, w2) =>
      (vocab.get(w1), vocab.get(w2)) match {
        case (Some((v1, _)), Some((v2, _))) =>
          val sim = cosineSimilarity(v1, computeNorm(v1), v2, computeNorm(v2))
          simWriter.println(f"$w1,$w2,$sim%.6f")
          println(f"  $w1%-15s <-> $w2%-15s: $sim%.4f")
        case _ =>
          println(s"  Missing: $w1 or $w2")
      }
    }
    simWriter.close()
    println("✓ word_similarities.csv written")

    // Word analogies
    println("\n" + "=" * 80)
    println("Task 3: Word Analogies")
    println("=" * 80)
    val analogyWriter = new PrintWriter("word_analogies.csv")
    analogyWriter.println("word_a,word_b,word_c,expected,pred_1,score_1,pred_2,score_2,pred_3,score_3")

    val analogyTests = Seq(("king", "man", "woman", "female"), ("city", "country", "paris", "amsterdam"), ("day", "night", "summer", "days"), ("love", "hate", "good", "nice")
    )

    analogyTests.foreach { case (a, b, c, expected) =>
      (vocab.get(a), vocab.get(b), vocab.get(c)) match {
        case (Some((va, _)), Some((vb, _)), Some((vc, _))) =>
          val target = add(sub(va, vb), vc)
          val targetNorm = computeNorm(target)
          val results = normalizedVocab
            .filterNot(nw => Set(a, b, c).contains(nw.word))
            .map(nw => (nw.word, cosineSimilarity(target, targetNorm, nw.vec, nw.norm)))
            .sortBy(-_._2)
            .take(3)

          val preds = results.map(_._1).mkString(",")
          val scores = results.map(p => f"${p._2}%.6f").mkString(",")
          analogyWriter.println(s"$a,$b,$c,$expected,$preds,$scores")
          println(s"  $a - $b + $c => ${results.head._1} (expected: $expected)")
        case _ =>
          println(s"  Missing words for analogy: $a - $b + $c")
      }
    }
    analogyWriter.close()
    println("✓ word_analogies.csv written")

    println("\n" + "=" * 80)
    println("Post-processing Complete!")
    println("=" * 80)
    println("Generated files:")
    println("  ✓ vocabulary.csv (full embeddings)")
    println("  ✓ nearest_neighbors.csv")
    println("  ✓ word_similarities.csv")
    println("  ✓ word_analogies.csv")
    println("=" * 80)
  }
}
