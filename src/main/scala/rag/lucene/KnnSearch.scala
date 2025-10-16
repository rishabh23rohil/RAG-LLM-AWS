package rag.lucene

import org.apache.lucene.store.FSDirectory
import org.apache.lucene.index.{DirectoryReader, MultiReader}
import org.apache.lucene.search.{IndexSearcher, KnnVectorQuery}
import org.apache.lucene.document.Document
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._
import rag.OllamaClient

object KnnSearch {
  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("usage: rag.lucene.KnnSearch <indexRootDir> <queryText> [k] [model]")
      System.err.println("examples:")
      System.err.println("  ... KnnSearch lucene-index neural networks and attention mechanisms")
      System.err.println("  ... KnnSearch lucene-index neural nets 10 default")
      System.exit(1)
    }

    val root  = args(0)
    val rest  = args.drop(1).toList

    val defaultK      = 5
    val defaultModel  = "default"

    // If we see ... <query words> <k> <model>   -> use both
    // Else if we see ... <query words> <k>      -> use k only
    // Else                                       -> all tokens are query
    val (k, model, qWords) =
      if (rest.length >= 2 && rest(rest.length - 2).forall(_.isDigit))
        (rest(rest.length - 2).toInt, rest.last, rest.dropRight(2))
      else if (rest.nonEmpty && rest.last.forall(_.isDigit))
        (rest.last.toInt, defaultModel, rest.dropRight(1))
      else
        (defaultK, defaultModel, rest)

    val qtext = qWords.mkString(" ")

    val rootPath = Paths.get(root)
    if (!Files.exists(rootPath)) {
      System.err.println(s"Index root not found: $root")
      System.exit(2)
    }

    val shardDirs =
      Files.list(rootPath).iterator().asScala
        .filter(p => Files.isDirectory(p) && p.getFileName.toString.startsWith("shard-"))
        .toList
        .sortBy(_.getFileName.toString)

    if (shardDirs.isEmpty) {
      System.err.println(s"No shard-* directories under: $root")
      System.exit(3)
    }

    val readers = shardDirs.map { p =>
      val dir = FSDirectory.open(p)
      DirectoryReader.open(dir)
    }

    val multi     = new MultiReader(readers*)
    val searcher  = new IndexSearcher(multi)
    val client    = new OllamaClient()

    val outVecs = client.embed(Vector(qtext), model)
    val emb: Array[Float] =
      if (outVecs.nonEmpty) outVecs.head
      else {
        System.err.println("Empty embedding returned; cannot run kNN query.")
        System.exit(4)
        Array.emptyFloatArray
      }

    val q   = new KnnVectorQuery("vec", emb, k)
    val top = searcher.search(q, k)

    println(s"Query: '$qtext'  k=$k  model=$model")
    println(s"Hits: ${top.scoreDocs.length}")
    top.scoreDocs.zipWithIndex.foreach { case (sd, i) =>
      val d: Document = searcher.doc(sd.doc)
      val docId   = Option(d.get("doc_id")).getOrElse("<no doc_id>")
      val chunkId = Option(d.get("chunk_id")).getOrElse("<no chunk_id>")
      val text    = Option(d.get("text")).getOrElse("").replaceAll("\\s+", " ").take(160)
      println(f"#${i+1}%2d  score=${sd.score}%1.5f  doc=$docId  chunk=$chunkId  text=\"$text\"")
    }

    multi.close()
    readers.foreach(_.close())
  }
}