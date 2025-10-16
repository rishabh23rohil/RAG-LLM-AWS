package rag.stats

import com.typesafe.config.ConfigFactory
import org.apache.lucene.store.FSDirectory
import org.apache.lucene.index.{DirectoryReader, MultiReader}
import org.apache.lucene.search.{IndexSearcher, KnnVectorQuery}
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._
import sttp.client3.*
import io.circe.*, io.circe.generic.semiauto.*
import rag.{OllamaClient}
import java.nio.file.StandardOpenOption
import org.slf4j.LoggerFactory

object SimilarityEval:
  private val log = LoggerFactory.getLogger(getClass)

  // Small probe sets
  private val simPairs = Vector(
    "cat" -> "dog",
    "king" -> "queen",
    "java" -> "scala",
    "linux" -> "windows",
    "bug" -> "issue"
  )

  // Analogy a:b :: c:?
  private val analogies = Vector(
    ("king", "man", "woman", "queen"),
    ("paris", "france", "italy", "rome"),
    ("teacher", "school", "hospital", "doctor")
  )

  private def cos(a: Array[Float], b: Array[Float]): Double =
    val dot = a.indices.foldLeft(0.0)((s,i) => s + a(i)*b(i))
    val na  = math.sqrt(a.foldLeft(0.0)((s,x)=> s + x*x))
    val nb  = math.sqrt(b.foldLeft(0.0)((s,x)=> s + x*x))
    if na==0 || nb==0 then 0.0 else dot/(na*nb)

  def main(args: Array[String]): Unit =
    val conf   = ConfigFactory.load()
    val root   = if args.nonEmpty then args(0) else conf.getString("app.indexDir")
    val outCsv = if args.length>=2 then args(1) else conf.getString("app.eval.outputCsv")
    val model  = conf.getString("app.model")

    val client = new OllamaClient()
    def E(tokens: Vector[String]): Vector[Array[Float]] = client.embed(tokens, model)

    val simRows = simPairs.map { case (a,b) =>
      val ev = E(Vector(a,b))
      val s  = if ev.length==2 then cos(ev(0), ev(1)) else 0.0
      (a,b,f"$s%.4f")
    }

    val analogyRows = analogies.map { case (a,b,c,expected) =>
      val ev = E(Vector(a,b,c))
      val score =
        if ev.length==3 then
          val vec = ev(0).indices.map(i => ev(0)(i) - ev(1)(i) + ev(2)(i)).map(_.toFloat).toArray
          // find NN among (expected, a,b,c) for simplicity; in a fuller eval you’d search vocab
          val cand = E(Vector(expected, a, b, c))
          val scores = cand.map(cos(vec,_))
          val bestIx = scores.indices.maxBy(scores)
          if bestIx==0 then 1.0 else 0.0
        else 0.0
      (s"$a-$b+$c", expected, f"$score%.1f")
    }

    val path = Paths.get(outCsv)
    Files.createDirectories(path.getParent)
    val header = "type,left,right_or_formula,score_or_expected\n"
    Files.writeString(path, header, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    simRows.foreach { case (l,r,s) =>
      Files.writeString(path, s"similarity,$l,$r,$s\n", StandardOpenOption.APPEND)
    }
    analogyRows.foreach { case (formula,exp,ok) =>
      Files.writeString(path, s"analogy,$formula,$exp,$ok\n", StandardOpenOption.APPEND)
    }
    log.info(s"Wrote eval CSV -> $outCsv")
