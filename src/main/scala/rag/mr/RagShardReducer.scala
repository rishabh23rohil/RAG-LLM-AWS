package rag.mr

import org.apache.hadoop.io.{IntWritable, Text}
import org.apache.hadoop.mapreduce.Reducer
import org.apache.lucene.document._
import org.apache.lucene.index._
import org.apache.lucene.store.FSDirectory
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.index.VectorSimilarityFunction
import java.nio.file.{Files, Path}
import io.circe.parser.parse

class RagShardReducer extends Reducer[IntWritable, Text, Text, Text] {
  override def reduce(key: IntWritable, values: java.lang.Iterable[Text],
                      ctx: Reducer[IntWritable,Text,Text,Text]#Context): Unit = {

    val shard = key.get
    val local: Path = Files.createTempDirectory(s"lucene-shard-$shard")
    val writer = new IndexWriter(
      FSDirectory.open(local),
      new IndexWriterConfig(new StandardAnalyzer())
    )

    val it = values.iterator()
    while (it.hasNext) {
      val rec = parse(it.next().toString).toOption.get.hcursor
      val doc = new Document()
      doc.add(new StringField("doc_id",   rec.get[String]("doc_id").toOption.get, Field.Store.YES))
      doc.add(new StringField("chunk_id", rec.get[Int]("chunk_id").toOption.get.toString, Field.Store.YES))
      doc.add(new TextField("text",       rec.get[String]("text").toOption.get, Field.Store.YES))
      val vec = rec.get[Vector[Float]]("vec").toOption.get.toArray
      doc.add(new KnnFloatVectorField("vec", vec, VectorSimilarityFunction.COSINE))
      writer.addDocument(doc)
    }
    writer.commit(); writer.close()

    // In local mode we just tell you where the shard index was written
    ctx.write(new Text(s"shard=$shard"), new Text(s"localIndex=${local.toAbsolutePath}"))
  }
}