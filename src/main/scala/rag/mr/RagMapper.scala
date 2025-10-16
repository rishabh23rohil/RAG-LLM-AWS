package rag.mr

import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import java.nio.file.{Paths, Files}
import rag.{Pdfs, Chunker, OllamaClient}
import io.circe.syntax._
import io.circe.Json

class RagMapper extends Mapper[LongWritable, Text, IntWritable, Text] {
  private val client = new OllamaClient()
  private val model  = sys.env.getOrElse("EMBED_MODEL", "mxbai-embed-large")

  override def map(key: LongWritable, v: Text, ctx: Mapper[LongWritable,Text,IntWritable,Text]#Context): Unit = {
    val pStr = v.toString.trim
    if (pStr.isEmpty) return
    val p = Paths.get(pStr)
    if (!Files.exists(p)) return

    val docId  = p.getFileName.toString
    val text   = Pdfs.readText(p)
    val chunks = Chunker.split(text)
    val vecs   = client.embed(chunks.toVector, model) // Vector[Array[Float]]

    val shard  = math.abs(docId.hashCode) % math.max(ctx.getNumReduceTasks, 1)
    chunks.zip(vecs).zipWithIndex.foreach { case ((c, e), ix) =>
      val json = Map(
        "doc_id"   -> Json.fromString(docId),
        "chunk_id" -> Json.fromInt(ix),
        "text"     -> Json.fromString(c),
        "vec"      -> Json.fromValues(e.toVector.map(x => Json.fromFloatOrNull(x)))
      ).asJson.noSpaces
      ctx.write(new IntWritable(shard), new Text(json))
    }
  }
}