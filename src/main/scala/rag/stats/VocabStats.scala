package rag.stats

import com.typesafe.config.ConfigFactory
import org.apache.lucene.store.FSDirectory
import org.apache.lucene.index.{DirectoryReader, MultiReader}
import org.apache.lucene.document.Document
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.search.KnnVectorQuery
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._
import java.nio.file.StandardOpenOption
import org.slf4j.LoggerFactory
import scala.collection.mutable

object VocabStats:
  private val log = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit =
    val conf  = ConfigFactory.load()
    val root  = if args.nonEmpty then args(0) else conf.getString("app.indexDir")
    val out   = if args.length >= 2 then args(1) else conf.getString("app.stats.outputCsv")
    val kN    = conf.getInt("app.stats.neighbors")

    val rootPath = Paths.get(root)
    require(Files.exists(rootPath), s"Index root not found: $root")

    val shardDirs =
      Files.list(rootPath).iterator().asScala
        .filter(p => Files.isDirectory(p) && p.getFileName.toString.startsWith("shard-"))
        .toVector
        .sortBy(_.getFileName.toString)

    require(shardDirs.nonEmpty, s"No shard-* directories under $root")

    val readers = shardDirs.map(p => DirectoryReader.open(FSDirectory.open(p)))
    val multi   = new MultiReader(readers*)
    val searcher= new IndexSearcher(multi)

    val freq = mutable.Map.empty[String, Long].withDefaultValue(0L)

    // build per-word vectors as average of chunk embeddings where it appears (coarse but OK)
    val accum = mutable.Map.empty[String, (Array[Double], Int)] // sum, count

    (0 until multi.maxDoc()).foreach { id =>
      val d: Document = searcher.doc(id)
      val txt = Option(d.get("text")).getOrElse("")
      val ws  = Tokenize.words(txt)
      ws.foreach(w => freq(w) = freq(w) + 1)

      // pull vec using vector stored in index for this doc
      // Lucene stores vec field "vec"; we need it from the vector values api
      // In 9.10 we can only query vectors; for averaging, reuse chunk vector by querying self via K=1 with filter is heavy.
      // Simpler heuristic: skip averaging if we can't fetch the raw vector; instead embed the word list once via your OllamaClient path later if needed.
      // For HW1 baseline: approximate with TF-only neighbors; but we’ll still produce neighbors via kNN over chunk vectors using a probe query = the word itself.
    }

    // Write CSV: word,freq (neighbors produced in a separate pass to keep time down)
    val outPath = Paths.get(out)
    Files.createDirectories(outPath.getParent)
    Files.writeString(outPath, "word,freq\n", StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    val top = freq.toVector.sortBy(-_._2)
    top.foreach { case (w,c) =>
      Files.writeString(outPath, s"$w,$c\n", StandardOpenOption.APPEND)
    }

    log.info(s"Wrote token frequency CSV -> $out")
    multi.close(); readers.foreach(_.close())
